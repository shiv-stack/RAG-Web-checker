"""scrapers.py
Web scraping, metadata extraction, WHOIS lookup, and tech stack detection.
"""

import re
import socket
from typing import Any, Dict, List
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
import whois

# ---------------------------------------------------------------------------
# Force IPv4 in urllib3 to prevent [Errno 101] Network is unreachable on Render,
# while preserving standard socket DNS resolution for APIs and proxies.
# ---------------------------------------------------------------------------
try:
    import urllib3.util.connection as urllib3_cn
    urllib3_cn.HAS_IPV6 = False
    urllib3_cn.allowed_gai_family = lambda: urllib3_cn.socket.AF_INET
except Exception:
    pass


# Default headers to mimic a modern browser and prevent 403/timeout responses
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}


def sanitize_url(raw_url: str) -> str:
    """Ensure URL has a valid HTTP/HTTPS scheme and clean structure."""
    raw_url = raw_url.strip()
    if not raw_url.startswith(("http://", "https://")):
        raw_url = "https://" + raw_url
    return raw_url


def extract_domain(url: str) -> str:
    """Extract root domain name from a URL."""
    parsed = urlparse(url)
    netloc = parsed.netloc or parsed.path
    # Strip port if present
    netloc = netloc.split(":")[0]
    # Strip leading 'www.'
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def fetch_whois_data(domain: str) -> Dict[str, Any]:
    """
    Perform WHOIS lookup for the root domain.
    Gracefully handles timeouts, registry blocks, and missing fields.
    """
    whois_info = {
        "domain_name": domain,
        "registrar": "Not Available",
        "creation_date": "Not Available",
        "expiration_date": "Not Available",
        "country": "Not Available",
        "org": "Not Available",
        "status": "Lookup failed or unavailable",
    }

    try:
        # Prevent WHOIS from hanging on cloud platforms that block outbound port 43
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(4)
        try:
            query_res = whois.whois(domain)
        finally:
            socket.setdefaulttimeout(old_timeout)

        if query_res:
            # Registrar
            if query_res.get("registrar"):
                whois_info["registrar"] = str(query_res.get("registrar"))

            # Creation date
            created = query_res.get("creation_date")
            if isinstance(created, list) and created:
                created = created[0]
            if created:
                whois_info["creation_date"] = (
                    created.strftime("%Y-%m-%d")
                    if hasattr(created, "strftime")
                    else str(created)
                )

            # Expiration date
            expired = query_res.get("expiration_date")
            if isinstance(expired, list) and expired:
                expired = expired[0]
            if expired:
                whois_info["expiration_date"] = (
                    expired.strftime("%Y-%m-%d")
                    if hasattr(expired, "strftime")
                    else str(expired)
                )

            # Country & Organization
            if query_res.get("country"):
                whois_info["country"] = str(query_res.get("country"))
            if query_res.get("org"):
                whois_info["org"] = str(query_res.get("org"))

            whois_info["status"] = "Success"
    except Exception as e:
        whois_info["status"] = f"Unavailable ({type(e).__name__})"

    return whois_info


def _generate_candidate_urls(clean_url: str) -> List[str]:
    """Generate ordered list of fallback URLs (https -> www -> http)."""
    parsed = urlparse(clean_url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc or parsed.path
    path = parsed.path if parsed.netloc else ""
    query = f"?{parsed.query}" if parsed.query else ""
    suffix = f"{path}{query}"

    candidates = [clean_url]

    # Try toggling 'www.'
    if netloc.startswith("www."):
        alt_netloc = netloc[4:]
    else:
        alt_netloc = f"www.{netloc}"
    candidates.append(f"{scheme}://{alt_netloc}{suffix}")

    # Fallback to plain http:// if https:// drops connections
    if scheme == "https":
        candidates.append(f"http://{netloc}{suffix}")
        candidates.append(f"http://{alt_netloc}{suffix}")

    # Deduplicate while preserving order
    return list(dict.fromkeys(candidates))


def _fetch_via_proxy(clean_url: str, domain: str) -> Optional[Dict[str, Any]]:
    """
    Fallback reader proxy to bypass datacenter IP blocks,
    regional firewalls, and anti-bot protection when deployed on cloud platforms.
    """
    # 1. Try Jina Reader
    try:
        jina_url = f"https://r.jina.ai/{clean_url}"
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            "Accept": "text/plain, text/html",
            "X-No-Cache": "true",
        }
        res = requests.get(jina_url, headers=headers, timeout=(6, 15))
        if res.status_code == 200 and len(res.text.strip()) > 80:
            lines = [l.strip() for l in res.text.splitlines() if l.strip()]
            title = lines[0][:100] if lines else f"{domain}"
            desc = " ".join(lines[1:5])[:250] if len(lines) > 1 else "Extracted page content"

            return {
                "url": clean_url,
                "domain": domain,
                "status_code": 200,
                "title": title,
                "description": desc,
                "author": "Not specified",
                "copyright": "None detected",
                "og_image": None,
                "raw_html": f"<html><head><title>{title}</title></head><body><main>{res.text}</main></body></html>",
                "headers": dict(res.headers),
                "error": None,
            }
    except Exception:
        pass

    # 2. Try AllOrigins public proxy
    try:
        ao_url = f"https://api.allorigins.win/raw?url={clean_url}"
        res = requests.get(ao_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=(6, 15))
        if res.status_code == 200 and len(res.text.strip()) > 80:
            soup = BeautifulSoup(res.text, "html.parser")
            title_tag = soup.find("title")
            title = title_tag.string.strip() if title_tag and title_tag.string else domain
            return {
                "url": clean_url,
                "domain": domain,
                "status_code": 200,
                "title": title,
                "description": "Fetched via redundant proxy",
                "author": "Not specified",
                "copyright": "None detected",
                "og_image": None,
                "raw_html": res.text,
                "headers": dict(res.headers),
                "error": None,
            }
    except Exception:
        pass

    return None


def extract_metadata_and_html(url: str, timeout: int = 12) -> Dict[str, Any]:
    """
    Fetch raw HTML from target URL and extract structured metadata:
    Title, Meta description, Author/Publisher, OpenGraph tags, and copyright notice.
    Automatically handles datacenter IP blocking, SSL issues, and cloud firewall timeouts.
    """
    clean_url = sanitize_url(url)
    domain = extract_domain(clean_url)
    result = {
        "url": clean_url,
        "domain": domain,
        "status_code": None,
        "title": "Untitled",
        "description": "No description found",
        "author": "Not specified",
        "copyright": "None detected",
        "og_image": None,
        "raw_html": "",
        "headers": {},
        "error": None,
    }

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=1)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    response = None
    last_error = None
    successful_url = clean_url

    # 1. Primary Direct Attempt
    try:
        res = session.get(
            clean_url,
            headers=DEFAULT_HEADERS,
            timeout=(5, 10),
            allow_redirects=True,
            verify=True,
        )
        if res.status_code < 400:
            response = res
            successful_url = res.url or clean_url
    except requests.exceptions.SSLError:
        try:
            res = session.get(
                clean_url,
                headers=DEFAULT_HEADERS,
                timeout=(5, 10),
                allow_redirects=True,
                verify=False,
            )
            if res.status_code < 400:
                response = res
                successful_url = res.url or clean_url
        except Exception as e:
            last_error = e
    except Exception as e:
        last_error = e

    # 2. If direct attempt failed or timed out (cloud datacenter block), try Proxy Fallback
    if response is None:
        proxy_result = _fetch_via_proxy(clean_url, domain)
        if proxy_result:
            return proxy_result

        # 3. Secondary candidate attempt (e.g. toggling www)
        candidates = _generate_candidate_urls(clean_url)
        for target in candidates:
            if target == clean_url:
                continue
            try:
                res = session.get(
                    target,
                    headers=DEFAULT_HEADERS,
                    timeout=(4, 8),
                    allow_redirects=True,
                    verify=False,
                )
                if res.status_code < 400:
                    response = res
                    successful_url = res.url or target
                    break
            except Exception as e:
                last_error = e
                continue

    # If all attempts failed
    if response is None:
        result["error"] = (
            f"Failed to retrieve {clean_url}: {str(last_error)}. "
            "The target website is completely unresponsive or refusing all connections."
        )
        return result

    result["url"] = successful_url
    result["domain"] = extract_domain(successful_url)
    result["status_code"] = response.status_code
    result["headers"] = dict(response.headers)
    result["raw_html"] = response.text

    try:
        soup = BeautifulSoup(response.text, "html.parser")

        # 1. Page Title
        title_tag = soup.find("title")
        og_title = soup.find("meta", property="og:title")
        if title_tag and title_tag.string:
            result["title"] = title_tag.string.strip()
        elif og_title and og_title.get("content"):
            result["title"] = og_title.get("content").strip()

        # 2. Meta Description
        meta_desc = (
            soup.find("meta", attrs={"name": "description"})
            or soup.find("meta", attrs={"property": "og:description"})
            or soup.find("meta", attrs={"name": "twitter:description"})
        )
        if meta_desc and meta_desc.get("content"):
            result["description"] = meta_desc.get("content").strip()

        # 3. Author / Creator / Publisher
        meta_author = (
            soup.find("meta", attrs={"name": "author"})
            or soup.find("meta", attrs={"property": "article:author"})
            or soup.find("meta", attrs={"name": "creator"})
            or soup.find("meta", attrs={"name": "publisher"})
        )
        if meta_author and meta_author.get("content"):
            result["author"] = meta_author.get("content").strip()

        # 4. OpenGraph Image
        og_img = soup.find("meta", property="og:image")
        if og_img and og_img.get("content"):
            result["og_image"] = og_img.get("content")

        # 5. Copyright / Footer heuristic
        footer = soup.find("footer")
        search_scope = footer.get_text() if footer else response.text[-5000:]
        copyright_match = re.search(
            r"(?:©|&copy;|copyright)\s*(?:20\d\d|19\d\d)?(?:\s*-\s*20\d\d)?\s*([^,\.\n\<\>]{3,50})",
            search_scope,
            re.IGNORECASE,
        )
        if copyright_match:
            result["copyright"] = copyright_match.group(0).strip()
    except Exception as e:
        # Parsing error shouldn't fail the entire response
        pass

    return result


def detect_tech_stack(soup: BeautifulSoup, headers: Dict[str, str], raw_html: str) -> Dict[str, List[str]]:
    """
    Detect technologies used by inspecting:
    1. HTTP Response Headers (Server, X-Powered-By, CDNs)
    2. Meta Generators and script/link tags
    3. Global JavaScript variables and DOM signatures
    """
    tech_stack: Dict[str, List[str]] = {
        "CMS & Platforms": [],
        "Frontend Frameworks": [],
        "Styling & UI": [],
        "Hosting & CDN": [],
        "Analytics & Tags": [],
        "Server & Backend": [],
    }

    raw_html_lower = raw_html.lower()
    headers_lower = {k.lower(): v.lower() for k, v in headers.items()}
    scripts_src = " ".join([s.get("src", "").lower() for s in soup.find_all("script") if s.get("src")])
    links_href = " ".join([l.get("href", "").lower() for l in soup.find_all("link") if l.get("href")])

    # 1. CMS & Platforms
    generator = soup.find("meta", attrs={"name": "generator"})
    gen_content = generator.get("content", "").lower() if generator else ""

    if "wordpress" in gen_content or "wp-content" in raw_html_lower or "wp-includes" in raw_html_lower:
        tech_stack["CMS & Platforms"].append("WordPress")
    if "shopify" in raw_html_lower or "cdn.shopify.com" in scripts_src or "window.shopify" in raw_html_lower:
        tech_stack["CMS & Platforms"].append("Shopify")
    if "webflow" in gen_content or "data-wf-site" in raw_html_lower or "webflow.com" in scripts_src:
        tech_stack["CMS & Platforms"].append("Webflow")
    if "squarespace" in raw_html_lower or "static1.squarespace.com" in scripts_src:
        tech_stack["CMS & Platforms"].append("Squarespace")
    if "wix.com" in scripts_src or "wix" in gen_content:
        tech_stack["CMS & Platforms"].append("Wix")
    if "ghost" in gen_content or "ghost" in raw_html_lower:
        tech_stack["CMS & Platforms"].append("Ghost")
    if "drupal" in gen_content or "drupal" in raw_html_lower:
        tech_stack["CMS & Platforms"].append("Drupal")
    if "strapi" in raw_html_lower:
        tech_stack["CMS & Platforms"].append("Strapi")

    # 2. Frontend Frameworks
    if "__next_data__" in raw_html_lower or "_next/static" in scripts_src or "_next/static" in raw_html_lower:
        tech_stack["Frontend Frameworks"].append("Next.js")
        tech_stack["Frontend Frameworks"].append("React")
    elif "react" in scripts_src or "reactroot" in raw_html_lower or "_reactlistening" in raw_html_lower:
        tech_stack["Frontend Frameworks"].append("React")

    if "__nuxt__" in raw_html_lower or "_nuxt/" in scripts_src:
        tech_stack["Frontend Frameworks"].append("Nuxt.js")
        tech_stack["Frontend Frameworks"].append("Vue.js")
    elif "vue" in scripts_src or "data-v-" in raw_html_lower:
        tech_stack["Frontend Frameworks"].append("Vue.js")

    if "ng-version" in raw_html_lower or "ng-app" in raw_html_lower or "angular" in scripts_src:
        tech_stack["Frontend Frameworks"].append("Angular")

    if "svelte" in raw_html_lower or "sveltekit" in raw_html_lower:
        tech_stack["Frontend Frameworks"].append("Svelte")

    if "astro" in raw_html_lower:
        tech_stack["Frontend Frameworks"].append("Astro")

    if "jquery" in scripts_src or "jquery.min.js" in raw_html_lower:
        tech_stack["Frontend Frameworks"].append("jQuery")

    # 3. Styling & UI
    if "tailwind" in links_href or "tailwindcss" in raw_html_lower or re.search(r'class="[^"]*(?:flex|grid|hidden|p-\d|m-\d|text-|bg-)[^"]*"', raw_html):
        # Additional validation to avoid false positives
        if "tailwind" in raw_html_lower or "tw-" in raw_html_lower or "flex " in raw_html_lower:
            tech_stack["Styling & UI"].append("Tailwind CSS")

    if "bootstrap" in links_href or "bootstrap" in scripts_src or "bootstrap.min.css" in raw_html_lower:
        tech_stack["Styling & UI"].append("Bootstrap")

    if "font-awesome" in links_href or "fontawesome" in links_href:
        tech_stack["Styling & UI"].append("Font Awesome")

    # 4. Hosting & CDN
    if "cloudflare" in headers_lower.get("server", "") or "cf-ray" in headers_lower or "cloudflare" in scripts_src:
        tech_stack["Hosting & CDN"].append("Cloudflare")

    if "x-vercel-id" in headers_lower or "x-vercel-cache" in headers_lower or "vercel" in headers_lower.get("server", ""):
        tech_stack["Hosting & CDN"].append("Vercel")

    if "x-amz-cf-id" in headers_lower or "cloudfront" in headers_lower.get("via", ""):
        tech_stack["Hosting & CDN"].append("AWS CloudFront")

    if "x-github-request-id" in headers_lower or "github.io" in raw_html_lower:
        tech_stack["Hosting & CDN"].append("GitHub Pages")

    if "x-nf-request-id" in headers_lower or "netlify" in headers_lower.get("server", ""):
        tech_stack["Hosting & CDN"].append("Netlify")

    if "fastly" in headers_lower.get("via", "") or "x-fastly-request-id" in headers_lower:
        tech_stack["Hosting & CDN"].append("Fastly")

    # 5. Server & Backend hints
    server_header = headers.get("Server") or headers.get("server")
    if server_header:
        tech_stack["Server & Backend"].append(server_header)
    powered_by = headers.get("X-Powered-By") or headers.get("x-powered-by")
    if powered_by:
        tech_stack["Server & Backend"].append(f"Powered by: {powered_by}")

    # 6. Analytics & Tags
    if "googletagmanager.com" in scripts_src or "gtag(" in raw_html_lower:
        tech_stack["Analytics & Tags"].append("Google Tag Manager / Analytics")
    if "connect.facebook.net" in scripts_src or "fbq(" in raw_html_lower:
        tech_stack["Analytics & Tags"].append("Meta Pixel")
    if "hotjar.com" in scripts_src:
        tech_stack["Analytics & Tags"].append("Hotjar")
    if "segment.com" in scripts_src or "analytics.js" in scripts_src:
        tech_stack["Analytics & Tags"].append("Segment")
    if "intercom" in scripts_src or "widget.intercom.io" in scripts_src:
        tech_stack["Analytics & Tags"].append("Intercom")

    # Deduplicate entries while preserving order
    for category in tech_stack:
        tech_stack[category] = list(dict.fromkeys(tech_stack[category]))

    return tech_stack


def analyze_target_website(url: str) -> Dict[str, Any]:
    """
    Master runner: Fetches page, extracts metadata, WHOIS data, and detects tech stack.
    """
    clean_url = sanitize_url(url)
    domain = extract_domain(clean_url)

    # 1. Fetch metadata & HTML
    meta_info = extract_metadata_and_html(clean_url)

    if meta_info.get("error"):
        return {
            "success": False,
            "error": meta_info["error"],
            "url": clean_url,
            "domain": domain,
        }

    soup = BeautifulSoup(meta_info["raw_html"], "html.parser")

    # 2. Detect tech stack
    tech = detect_tech_stack(
        soup=soup,
        headers=meta_info.get("headers", {}),
        raw_html=meta_info.get("raw_html", ""),
    )

    # 3. WHOIS lookup
    whois_info = fetch_whois_data(domain)

    return {
        "success": True,
        "url": clean_url,
        "domain": domain,
        "metadata": {
            "title": meta_info.get("title"),
            "description": meta_info.get("description"),
            "author": meta_info.get("author"),
            "copyright": meta_info.get("copyright"),
            "og_image": meta_info.get("og_image"),
            "status_code": meta_info.get("status_code"),
        },
        "whois": whois_info,
        "tech_stack": tech,
        "raw_html": meta_info.get("raw_html", ""),
    }
