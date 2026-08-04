#!/usr/bin/env ruby

require 'date'
require 'pathname'
require 'time'
require 'yaml'

ROOT = Pathname(__dir__).join('..').expand_path

EXPECTED_URLS = {
  '2025-10-27-hello-world.md' => '/2025/10/27/hello-world/',
  '2025-11-03-Transformer-position-embedding1.md' => '/2025/11/03/Transformer-position-embedding1/',
  '2025-11-04-Transformer-position-embedding2.md' => '/2025/11/04/Transformer-position-embedding2/',
  '2025-11-05-论文阅读1-tpugraphs.md' => '/2025/11/05/论文阅读1-tpugraphs/',
  '2025-11-11-Transformer3.md' => '/2025/11/11/Transformer3/',
  '2026-07-14-fpga-cgra-systolic-array-report.md' => '/2026/07/14/fpga-cgra-systolic-array-report/'
}.freeze

posts = ROOT.join('_posts').children.select(&:file?)
actual_names = posts.map { |post| post.basename.to_s.force_encoding(Encoding::UTF_8) }.sort
expected_names = EXPECTED_URLS.keys.sort
abort "Unexpected post set: #{actual_names.inspect}" unless actual_names == expected_names

references = []

posts.each do |post|
  content = post.read(encoding: 'UTF-8')
  front_matter = content.match(/\A---\s*\n(.*?)\n---\s*\n/m)&.[](1)
  abort "Missing front matter: #{post}" unless front_matter

  metadata = YAML.safe_load(front_matter, permitted_classes: [Date, Time], aliases: true)
  abort "Missing title: #{post}" if metadata['title'].to_s.empty?
  abort "Missing description: #{post}" if metadata['description'].to_s.empty?

  unresponsive_images = content.lines.grep(/^!\[/).reject { |line| line.include?('{: .img-fluid }') }
  abort "Markdown images must be responsive: #{post}" unless unresponsive_images.empty?

  references.concat(content.scan(%r{\{\{ '/assets/img/posts/([^']+)' \| relative_url \}\}}).flatten)

  filename = post.basename.to_s.force_encoding(Encoding::UTF_8)
  expected_url = EXPECTED_URLS.fetch(filename)
  match = filename.match(/\A(\d{4})-(\d{2})-(\d{2})-(.+)\.md\z/)
  actual_url = "/#{match[1]}/#{match[2]}/#{match[3]}/#{match[4]}/"
  abort "URL changed for #{filename}: #{actual_url}" unless actual_url == expected_url
end

assets = ROOT.join('assets/img/posts').glob('**/*').select(&:file?).map do |asset|
  asset.relative_path_from(ROOT.join('assets/img/posts')).to_s.force_encoding(Encoding::UTF_8)
end

missing = references.uniq - assets
unused = assets - references.uniq
abort "Missing image assets: #{missing.join(', ')}" unless missing.empty?

puts "Migration valid: #{posts.length} posts, #{assets.length} images, all legacy URLs preserved"
puts "Retained but currently unreferenced: #{unused.join(', ')}" unless unused.empty?
