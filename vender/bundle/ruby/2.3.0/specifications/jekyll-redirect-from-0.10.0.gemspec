# -*- encoding: utf-8 -*-
# stub: jekyll-redirect-from 0.10.0 ruby lib

Gem::Specification.new do |s|
  s.name = "jekyll-redirect-from"
  s.version = "0.10.0"

  s.required_rubygems_version = Gem::Requirement.new(">= 0") if s.respond_to? :required_rubygems_version=
  s.require_paths = ["lib"]
  s.authors = ["Parker Moore"]
  s.date = "2016-03-16"
  s.description = "Seamlessly specify multiple redirection URLs for your pages and posts"
  s.email = ["parkrmoore@gmail.com"]
  s.homepage = "https://github.com/jekyll/jekyll-redirect-from"
  s.licenses = ["MIT"]
  s.rubygems_version = "2.5.1"
  s.summary = "Seamlessly specify multiple redirection URLs for your pages and posts"

  s.installed_by_version = "2.5.1" if s.respond_to? :installed_by_version

  if s.respond_to? :specification_version then
    s.specification_version = 4

    if Gem::Version.new(Gem::VERSION) >= Gem::Version.new('1.2.0') then
      s.add_runtime_dependency(%q<jekyll>, [">= 2.0"])
      s.add_development_dependency(%q<bundler>, ["~> 1.3"])
      s.add_development_dependency(%q<rake>, [">= 0"])
      s.add_development_dependency(%q<rspec>, [">= 0"])
      s.add_development_dependency(%q<jekyll-sitemap>, ["~> 0.8.1"])
    else
      s.add_dependency(%q<jekyll>, [">= 2.0"])
      s.add_dependency(%q<bundler>, ["~> 1.3"])
      s.add_dependency(%q<rake>, [">= 0"])
      s.add_dependency(%q<rspec>, [">= 0"])
      s.add_dependency(%q<jekyll-sitemap>, ["~> 0.8.1"])
    end
  else
    s.add_dependency(%q<jekyll>, [">= 2.0"])
    s.add_dependency(%q<bundler>, ["~> 1.3"])
    s.add_dependency(%q<rake>, [">= 0"])
    s.add_dependency(%q<rspec>, [">= 0"])
    s.add_dependency(%q<jekyll-sitemap>, ["~> 0.8.1"])
  end
end
