#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const baseUrl = "http://127.0.0.1:8765/paper/gdor_slam_briefing_ppt.html";
const output = "/tmp/gdor_briefing_ppt_check";

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: "/usr/bin/google-chrome",
  });
  const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
  const consoleErrors = [];
  const failedRequests = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("response", (response) => {
    if (response.status() >= 400) failedRequests.push({
      status: response.status(),
      url: response.url(),
    });
  });
  await page.goto(baseUrl, { waitUntil: "networkidle" });
  fs.mkdirSync(output, { recursive: true });
  const slideCount = await page.locator(".slide").count();
  const problems = [];
  for (let index = 0; index < slideCount; index += 1) {
    await page.evaluate((slideIndex) => show(slideIndex, false), index);
    await page.waitForTimeout(80);
    const result = await page.evaluate(() => {
      const slide = document.querySelector(".slide.active");
      const slideRect = slide.getBoundingClientRect();
      const children = [...slide.children].filter(
        (child) => !child.classList.contains("top")
      );
      const clippedElements = children
        .map((child) => {
          const rect = child.getBoundingClientRect();
          return {
            tag: child.tagName,
            className: child.className,
            bottom: Math.round(rect.bottom),
          };
        })
        .filter((item) => item.bottom > Math.round(slideRect.bottom));
      return {
        title: slide.dataset.title,
        clippedElements,
        brokenImages: [...slide.querySelectorAll("img")]
          .filter((image) => !image.complete || image.naturalWidth === 0)
          .map((image) => image.getAttribute("src")),
      };
    });
    if (
      result.clippedElements.length ||
      result.brokenImages.length
    ) {
      problems.push(result);
    }
    if ([0, 4, 9, 11, slideCount - 1].includes(index)) {
      await page.screenshot({
        path: path.join(output, `slide_${String(index + 1).padStart(2, "0")}.png`),
      });
    }
  }
  const report = {
    slides: slideCount,
    problems,
    consoleErrors,
    failedRequests,
    screenshots: output,
  };
  console.log(JSON.stringify(report, null, 2));
  await browser.close();
  const relevantConsoleErrors = consoleErrors.filter(
    (message) => !message.includes("Failed to load resource")
  );
  const relevantRequests = failedRequests.filter(
    (request) => !request.url.endsWith("/favicon.ico")
  );
  if (problems.length || relevantConsoleErrors.length || relevantRequests.length) {
    process.exit(1);
  }
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
