# SamSz04's Blog

SamSz04 的个人主页与技术博客，基于 [al-folio](https://github.com/alshedivat/al-folio) v1.x 和 Jekyll 构建。

## 内容

- 个人主页、文章列表与 GitHub 仓库页
- 从旧 Hexo 站迁移的 6 篇文章与 53 个图片资源
- 保留旧站文章 URL：`/:year/:month/:day/:title/`
- 支持深色模式、全文搜索、数学公式、标签与分类归档

## 本地预览

推荐使用 Docker：

```bash
docker compose pull
docker compose up
```

然后访问 <http://localhost:8080>。

也可以使用本机 Ruby 环境：

```bash
bundle install
npm ci
bundle exec jekyll serve
```

然后访问 <http://localhost:4000>。

## 内容迁移

旧 Hexo 内容位于同级目录 `SamSz-blog-old/`。迁移脚本 `scripts/migrate_hexo_posts.rb` 会从旧站 `db.json` 恢复文章原始 Markdown，并将图片链接改写到 `assets/img/posts/`。重新迁移后请运行 `npx prettier '_posts/*.md' --write`。

## 部署

推送到 `SamSz04/SamSz04.github.io` 的 `main` 分支后，`.github/workflows/deploy.yml` 会构建站点并发布到 `gh-pages` 分支。GitHub Pages 的发布源应设置为 `gh-pages`。
