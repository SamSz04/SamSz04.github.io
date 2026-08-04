#!/usr/bin/env ruby

require 'date'
require 'fileutils'
require 'json'
require 'pathname'
require 'time'
require 'yaml'

ROOT = Pathname(__dir__).join('..').expand_path
DATABASE = ROOT.join('../SamSz-blog-old/db.json')
POSTS_DIR = ROOT.join('_posts')
ASSET_ROOT = ROOT.join('assets/img/posts')

DESCRIPTIONS = {
  'hello-world' => 'Hexo 博客的第一篇文章与快速入门记录。',
  'Transformer-position-embedding1' => '从自注意力的置换不变性出发，梳理绝对、相对与旋转位置编码的基本思路。',
  'Transformer-position-embedding2' => '系统梳理 RoPE 的原理、数学形式及其在 Transformer 中的实现逻辑。',
  'Transformer3' => '从 RoPE 的长度外推问题出发，解析 YaRN 的 NTK-by-parts 与注意力缩放。',
  '论文阅读1-tpugraphs' => 'TpuGraphs 论文阅读笔记：面向大型张量计算图的性能预测数据集。',
  'fpga-cgra-systolic-array-report' => '以深度学习硬件加速为背景，理解计算粒度、数据流与可重构性之间的取舍。'
}.freeze

def normalize_list(value)
  case value
  when nil
    []
  when Array
    value
  else
    [value]
  end
end

def split_front_matter(raw)
  match = raw.match(/\A---\s*\r?\n(.*?)\r?\n---\s*\r?\n/m)
  raise 'Post is missing YAML front matter' unless match

  [match[1], raw[match.end(0)..]]
end

def post_date(front_matter, fallback)
  raw_date = front_matter[/^date:\s*(.+?)\s*$/, 1]
  return Time.parse(fallback).getlocal('+08:00').strftime('%Y-%m-%d %H:%M:%S %z') unless raw_date

  "#{raw_date.sub(/\s+[+-]\d{4}\z/, '')} +0800"
end

def migrate_image_references(body, slug)
  asset_dir = ASSET_ROOT.join(slug)
  return body unless asset_dir.directory?

  asset_dir.children.select(&:file?).each do |asset|
    filename = asset.basename.to_s
    liquid_path = "{{ '/assets/img/posts/#{slug}/#{filename}' | relative_url }}"

    body = body.gsub("](#{filename})", "](#{liquid_path})")
    body = body.gsub(%(src="#{filename}"), %(src="#{liquid_path}"))
    body = body.gsub(%(src='#{filename}'), %(src="#{liquid_path}"))
  end

  body
end

def make_markdown_images_responsive(body)
  body.gsub(/^(!\[[^\n]*\]\([^\n]*\))$/) { "#{$1}{: .img-fluid }" }
end

database = JSON.parse(DATABASE.read)
posts = database.fetch('models').fetch('Post')

FileUtils.mkdir_p(POSTS_DIR)

posts.each do |post|
  slug = post.fetch('slug')
  front_matter, body = split_front_matter(post.fetch('raw'))
  metadata = YAML.safe_load(front_matter, permitted_classes: [Date, Time], aliases: true) || {}
  date = post_date(front_matter, post.fetch('date'))
  day = Date.parse(date)

  body = body.gsub("\r\n", "\n")
  body = body.gsub(/<!--\s*more\s*-->/i, '')
  body = body.gsub(/<span\s+id=["']more["']><\/span>/i, '')
  body = migrate_image_references(body, slug).strip
  body = make_markdown_images_responsive(body)

  output_metadata = {
    'layout' => 'post',
    'title' => post.fetch('title'),
    'date' => date,
    'description' => DESCRIPTIONS.fetch(slug),
    'tags' => normalize_list(metadata['tags']),
    'categories' => normalize_list(metadata['categories']),
    'related_posts' => slug != 'hello-world'
  }
  yaml = YAML.dump(output_metadata).sub(/\A---\s*\n/, '')
  filename = format('%<year>04d-%<month>02d-%<day>02d-%<slug>s.md', year: day.year, month: day.month, day: day.day, slug: slug)
  POSTS_DIR.join(filename).write("---\n#{yaml}---\n\n#{body}\n")
end

puts "Migrated #{posts.length} posts into #{POSTS_DIR}"
puts "Run: npx prettier '_posts/*.md' --write"
