#!/usr/bin/env python3
"""
实习僧爬虫 - Playwright版
- 不需要登录
- 常规翻页
- 进详情页爬取JD正文
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

from playwright.async_api import async_playwright

# 项目数据目录
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class ShiXiSengSpider:
    """实习僧爬虫"""

    BASE_URL = "https://www.shixiseng.com/interns"
    DETAIL_BASE = "https://www.shixiseng.com"

    def __init__(self, keyword: str, max_pages: int | None = None, headless: bool = True):
        self.keyword = keyword
        self.max_pages = max_pages  # None表示爬取全部
        self.headless = headless
        self.results = []

    def _build_list_url(self, page: int) -> str:
        params = {
            "page": page,
            "type": "intern",
            "keyword": self.keyword,
            "city": "全国",
        }
        return f"{self.BASE_URL}?{urlencode(params)}"

    async def _extract_list_items(self, page) -> list[dict]:
        """从列表页提取职位基本信息和详情页链接"""
        items = []
        wraps = await page.query_selector_all(".intern-wrap")

        for wrap in wraps:
            try:
                # 提取详情页链接
                link_el = await wrap.query_selector("a[href*='/intern/']")
                if not link_el:
                    continue
                href = await link_el.get_attribute("href")
                detail_url = href if href.startswith("http") else f"{self.DETAIL_BASE}{href}"

                # 提取文本信息（实习僧有字体反爬，部分文字可能为乱码，但链接和结构是稳定的）
                # 通过 outerHTML 中的 title 属性或 alt 属性可能获取真实文字
                title_text = await link_el.get_attribute("title") or ""

                # 尝试获取公司名
                company_el = await wrap.query_selector(".company-name, [class*='company']")
                company = ""
                if company_el:
                    company = await company_el.get_attribute("title") or await company_el.inner_text() or ""

                # 尝试获取城市、薪资等
                all_text = await wrap.inner_text()

                items.append({
                    "detail_url": detail_url.split("?")[0],  # 去掉查询参数
                    "title": title_text.strip(),
                    "company": company.strip(),
                    "list_text": all_text.strip(),
                })
            except Exception as e:
                print(f"  [WARN] 提取列表项失败: {e}")
                continue

        return items

    async def _extract_detail(self, page) -> dict:
        """从详情页提取JD正文"""
        result = {
            "job_description": "",
            "job_requirements": "",
            "raw_html_snippet": "",
        }

        try:
            # 等待页面主要内容加载
            await page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(1.5)  # 给JS渲染留时间

            # 策略1: 通过常见class/id找职位描述
            selectors = [
                ".job_detail",
                ".detail-content",
                ".position-desc",
                ".job-desc",
                "[class*='desc']",
                "[class*='detail']",
            ]

            for sel in selectors:
                try:
                    el = await page.query_selector(sel)
                    if el:
                        text = await el.inner_text()
                        if len(text) > 50:  # 过滤掉太短的
                            result["job_description"] = text.strip()
                            break
                except Exception:
                    continue

            # 策略2: 如果策略1没找到，尝试通过关键词定位
            if not result["job_description"]:
                html = await page.content()
                # 找包含"职位描述"或"岗位职责"的div/section
                patterns = [
                    r'(?:职位描述|岗位职责|岗位描述|工作描述)[\s\S]*?(<div[^>]*>[\s\S]{100,5000}?</div>)',
                    r'(?:职位描述|岗位职责|岗位描述|工作描述)[\s\S]*?(<section[^>]*>[\s\S]{100,5000}?</section>)',
                ]
                for pat in patterns:
                    match = re.search(pat, html, re.IGNORECASE)
                    if match:
                        snippet = match.group(1)
                        result["raw_html_snippet"] = snippet[:2000]
                        result["job_description"] = re.sub(r'<[^>]+>', ' ', snippet).strip()
                        break

            # 提取职位名称（详情页通常有清晰的title）
            title = await page.title()
            result["page_title"] = title

            # 尝试提取薪资、城市等信息
            page_text = await page.evaluate("() => document.body.innerText")
            result["page_text_sample"] = page_text[:500] if page_text else ""

        except Exception as e:
            print(f"  [WARN] 提取详情页失败: {e}")

        return result

    async def _has_next_page(self, page, current_page: int) -> bool:
        """检查是否有下一页"""
        # 检查页码输入框或下一页按钮
        try:
            # 找页码相关元素
            next_btn = await page.query_selector("a.next, .pagination .next, [class*='next']")
            if next_btn:
                classes = await next_btn.get_attribute("class") or ""
                if "disabled" in classes or "forbid" in classes:
                    return False
                return True

            # 或者看当前页码是否等于最大页码
            page_input = await page.query_selector("input[type='text'][class*='page']")
            if page_input:
                # 获取总页数提示
                page_text = await page.inner_text(".pagination, [class*='pagination']")
                if page_text:
                    total_match = re.search(r'(\d+)', page_text)
                    if total_match:
                        total = int(total_match.group(1))
                        return current_page < total
        except Exception:
            pass

        # 保守策略：尝试访问下一页，看是否有数据
        return True

    async def run(self) -> list[dict]:
        """执行爬取"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = await context.new_page()

            current_page = 1
            total_crawled = 0

            while True:
                list_url = self._build_list_url(current_page)
                print(f"\n[PAGE {current_page}] {list_url}")

                try:
                    await page.goto(list_url, wait_until="domcontentloaded", timeout=60000)
                    await asyncio.sleep(2)
                except Exception as e:
                    print(f"  [ERROR] 列表页加载失败: {e}")
                    break

                # 提取列表项
                items = await self._extract_list_items(page)
                print(f"  本页提取到 {len(items)} 条职位")

                if not items:
                    print("  本页无数据，结束爬取")
                    break

                # 逐个进入详情页
                for idx, item in enumerate(items, 1):
                    print(f"  [{idx}/{len(items)}] 详情页: {item['detail_url']}")
                    try:
                        detail_page = await context.new_page()
                        await detail_page.goto(item["detail_url"], wait_until="domcontentloaded", timeout=60000)
                        await asyncio.sleep(1.5)

                        detail_data = await self._extract_detail(detail_page)
                        item.update(detail_data)

                        # 添加元数据
                        item["search_keyword"] = self.keyword
                        item["crawl_time"] = datetime.now().isoformat()
                        item["page_no"] = current_page

                        self.results.append(item)
                        total_crawled += 1

                        await detail_page.close()
                        await asyncio.sleep(0.8)  # 礼貌延迟

                    except Exception as e:
                        print(f"    [ERROR] 详情页爬取失败: {e}")
                        continue

                # 检查终止条件
                if self.max_pages and current_page >= self.max_pages:
                    print(f"  达到最大页数限制 ({self.max_pages})，结束爬取")
                    break

                # 检查是否有下一页
                has_next = await self._has_next_page(page, current_page)
                if not has_next:
                    # 再验证一次：尝试点下一页或检查页面内容
                    try:
                        next_btn = await page.query_selector("a.next:not(.disabled):not(.forbid)")
                        if not next_btn:
                            # 看看有没有页码信息
                            page_info = await page.query_selector(".total, [class*='total']")
                            if page_info:
                                info_text = await page_info.inner_text()
                                print(f"  分页信息: {info_text}")
                    except Exception:
                        pass

                    # 如果当前页数据少于预期，也认为是最后一页
                    if len(items) < 5:
                        print("  无下一页，结束爬取")
                        break

                current_page += 1

            await browser.close()

        print(f"\n[完成] 关键词 '{self.keyword}' 共爬取 {total_crawled} 条")
        return self.results

    def save(self, filename: str | None = None):
        """保存结果到JSON"""
        if not filename:
            ts = datetime.now().strftime("%m%d_%H%M%S")
            filename = f"shixiseng_{self.keyword}_{ts}.json"
        filepath = DATA_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        print(f"[保存] 数据已保存至: {filepath}")
        return filepath


def crawl_twice():
    """
    分两次爬取：
    1. 搜索 'BI'，爬取全部数据
    2. 搜索 '数据'，最多爬30页
    """
    async def main():
        # 第一次：BI，全部数据
        print("=" * 50)
        print("【第一轮】搜索关键词: BI")
        print("=" * 50)
        spider1 = ShiXiSengSpider(keyword="BI", max_pages=None, headless=True)
        await spider1.run()
        spider1.save()

        # 第二次：数据，最多30页
        print("\n" + "=" * 50)
        print("【第二轮】搜索关键词: 数据 (上限30页)")
        print("=" * 50)
        spider2 = ShiXiSengSpider(keyword="数据", max_pages=30, headless=True)
        await spider2.run()
        spider2.save()

        print("\n" + "=" * 50)
        print("【全部完成】")
        print(f"BI: {len(spider1.results)} 条")
        print(f"数据: {len(spider2.results)} 条")
        print("=" * 50)

    asyncio.run(main())


if __name__ == "__main__":
    crawl_twice()
