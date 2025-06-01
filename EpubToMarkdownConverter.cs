using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml;
using System.Xml.Linq;

public class EpubToMarkdownConverter
{
    private readonly Dictionary<string, string> _manifestItems = new Dictionary<string, string>();
    private readonly List<string> _spineOrder = new List<string>();
    private string _tempDirectory;
    private string _opfPath;

    public bool IncludeLinks { get; init; } = false;

    public bool IncludeImages { get; init; } = false;

    public async Task<string> ConvertEpubToMarkdown(string epubFilePath)
    {
        try
        {
            // Create temporary directory for extraction
            _tempDirectory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
            Directory.CreateDirectory(_tempDirectory);

            // Extract EPUB contents
            ExtractEpub(epubFilePath);

            // Parse OPF file to get structure
            ParseOpfFile();

            // Convert content files to markdown
            var markdownContent = await ConvertContentToMarkdown();

            return markdownContent;
        }
        finally
        {
            // Clean up temporary directory
            if (Directory.Exists(_tempDirectory))
                Directory.Delete(_tempDirectory, true);
        }
    }

    private void ExtractEpub(string epubFilePath)
    {
        using (var archive = ZipFile.OpenRead(epubFilePath))
        {
            foreach (var entry in archive.Entries)
            {
                var destinationPath = Path.Combine(_tempDirectory, entry.FullName);
                var destinationDir = Path.GetDirectoryName(destinationPath);

                if (!Directory.Exists(destinationDir))
                    Directory.CreateDirectory(destinationDir);

                if (!entry.Name.Equals(""))
                    entry.ExtractToFile(destinationPath, true);
            }
        }
    }

    private void ParseOpfFile()
    {
        // Find container.xml to locate OPF file
        var containerPath = Path.Combine(_tempDirectory, "META-INF", "container.xml");
        if (!File.Exists(containerPath))
            throw new InvalidOperationException("Invalid EPUB: container.xml not found");

        var containerDoc = XDocument.Load(containerPath);
        var ns = containerDoc.Root.GetDefaultNamespace();
        var rootFileElement = containerDoc.Descendants(ns + "rootfile").FirstOrDefault();

        if (rootFileElement == null)
            throw new InvalidOperationException("Invalid EPUB: rootfile not found in container.xml");

        _opfPath = rootFileElement.Attribute("full-path")?.Value;
        if (string.IsNullOrEmpty(_opfPath))
            throw new InvalidOperationException("Invalid EPUB: OPF file path not found");

        // Parse OPF file
        var opfFullPath = Path.Combine(_tempDirectory, _opfPath);
        var opfDoc = XDocument.Load(opfFullPath);
        var opfNs = opfDoc.Root.GetDefaultNamespace();

        // Parse manifest
        var manifestElement = opfDoc.Descendants(opfNs + "manifest").FirstOrDefault();
        if (manifestElement != null)
        {
            foreach (var item in manifestElement.Descendants(opfNs + "item"))
            {
                var id = item.Attribute("id")?.Value;
                var href = item.Attribute("href")?.Value;
                if (!string.IsNullOrEmpty(id) && !string.IsNullOrEmpty(href))
                {
                    _manifestItems[id] = href;
                }
            }
        }

        // Parse spine
        var spineElement = opfDoc.Descendants(opfNs + "spine").FirstOrDefault();
        if (spineElement != null)
        {
            foreach (var itemref in spineElement.Descendants(opfNs + "itemref"))
            {
                var idref = itemref.Attribute("idref")?.Value;
                if (!string.IsNullOrEmpty(idref))
                {
                    _spineOrder.Add(idref);
                }
            }
        }
    }

    private async Task<string> ConvertContentToMarkdown()
    {
        var markdownBuilder = new StringBuilder();
        var opfDirectory = Path.GetDirectoryName(Path.Combine(_tempDirectory, _opfPath));

        foreach (var spineItemId in _spineOrder)
        {
            if (_manifestItems.TryGetValue(spineItemId, out var href))
            {
                var contentPath = Path.Combine(opfDirectory, href);
                if (File.Exists(contentPath))
                {
                    var content = await File.ReadAllTextAsync(contentPath);
                    var markdown = ConvertXhtmlToMarkdown(content);
                    markdownBuilder.AppendLine(markdown);
                    markdownBuilder.AppendLine(); // Add spacing between chapters
                }
            }
        }

        return markdownBuilder.ToString();
    }

    private string ConvertXhtmlToMarkdown(string xhtmlContent)
    {
        try
        {
            // Load XHTML content
            var doc = new XmlDocument();
            doc.LoadXml(xhtmlContent);

            // Extract body content
            var bodyNode = doc.SelectSingleNode("//body") ?? doc.DocumentElement;
            if (bodyNode == null) return string.Empty;

            return ConvertNodeToMarkdown(bodyNode);
        }
        catch (XmlException)
        {
            // If XML parsing fails, try to clean up HTML and convert
            return ConvertHtmlToMarkdown(xhtmlContent);
        }
    }

    private string ConvertNodeToMarkdown(XmlNode node)
    {
        var markdown = new StringBuilder();

        foreach (XmlNode child in node.ChildNodes)
        {
            switch (child.NodeType)
            {
                case XmlNodeType.Text:
                    markdown.Append(CleanText(child.InnerText));
                    break;

                case XmlNodeType.Element:
                    markdown.Append(ConvertElementToMarkdown(child));
                    break;
            }
        }

        return markdown.ToString();
    }

    private string ConvertElementToMarkdown(XmlNode element)
    {
        var tagName = element.LocalName.ToLower();
        var content = ConvertNodeToMarkdown(element);

        return tagName switch
        {
            "h1" => $"# {content.Trim()}\n\n",
            "h2" => $"## {content.Trim()}\n\n",
            "h3" => $"### {content.Trim()}\n\n",
            "h4" => $"#### {content.Trim()}\n\n",
            "h5" => $"##### {content.Trim()}\n\n",
            "h6" => $"###### {content.Trim()}\n\n",
            "p" => $"{content.Trim()}\n\n",
            "br" => "\n",
            "strong" or "b" => $"**{content}**",
            "em" or "i" => $"*{content}*",
            "u" => $"<u>{content}</u>",
            "code" => $"`{content}`",
            "pre" => $"```\n{content}\n```\n\n",
            "blockquote" => $"> {content.Replace("\n", "\n> ")}\n\n",
            "ul" => ConvertList(element, false),
            "ol" => ConvertList(element, true),
            "li" => content, // Handled by ConvertList
            "a" => this.IncludeLinks ? ConvertLink(element, content) : string.Empty,
            "img" => this.IncludeImages ? ConvertImage(element) : string.Empty,
            "hr" => "---\n\n",
            "div" or "span" => content,
            _ => content
        };
    }

    private string ConvertList(XmlNode listElement, bool isOrderedList)
    {
        var markdown = new StringBuilder();
        var items = listElement.SelectNodes("li");

        for (int i = 0; i < items.Count; i++)
        {
            var itemContent = ConvertNodeToMarkdown(items[i]).Trim();
            var prefix = isOrderedList ? $"{i + 1}. " : "- ";

            // Handle multi-line list items
            var lines = itemContent.Split('\n');
            markdown.AppendLine($"{prefix}{lines[0]}");

            for (int j = 1; j < lines.Length; j++)
            {
                if (!string.IsNullOrWhiteSpace(lines[j]))
                    markdown.AppendLine($"  {lines[j]}");
            }
        }

        markdown.AppendLine();
        return markdown.ToString();
    }

    private string ConvertLink(XmlNode linkElement, string content)
    {
        var href = linkElement.Attributes?["href"]?.Value;
        if (string.IsNullOrEmpty(href))
            return content;

        return $"[{content}]({href})";
    }

    private string ConvertImage(XmlNode imgElement)
    {
        var src = imgElement.Attributes?["src"]?.Value ?? "";
        var alt = imgElement.Attributes?["alt"]?.Value ?? "";
        var title = imgElement.Attributes?["title"]?.Value;

        if (!string.IsNullOrEmpty(title))
            return $"![{alt}]({src} \"{title}\")";

        return $"![{alt}]({src})";
    }

    private string ConvertHtmlToMarkdown(string htmlContent)
    {
        // Fallback HTML to Markdown conversion using regex
        var markdown = htmlContent;

        // Remove XML declaration and DOCTYPE
        markdown = Regex.Replace(markdown, @"<\?xml[^>]*\?>", "", RegexOptions.IgnoreCase);
        markdown = Regex.Replace(markdown, @"<!DOCTYPE[^>]*>", "", RegexOptions.IgnoreCase);

        // Convert headers
        markdown = Regex.Replace(markdown, @"<h1[^>]*>(.*?)</h1>", "# $1\n\n", RegexOptions.IgnoreCase | RegexOptions.Singleline);
        markdown = Regex.Replace(markdown, @"<h2[^>]*>(.*?)</h2>", "## $1\n\n", RegexOptions.IgnoreCase | RegexOptions.Singleline);
        markdown = Regex.Replace(markdown, @"<h3[^>]*>(.*?)</h3>", "### $1\n\n", RegexOptions.IgnoreCase | RegexOptions.Singleline);
        markdown = Regex.Replace(markdown, @"<h4[^>]*>(.*?)</h4>", "#### $1\n\n", RegexOptions.IgnoreCase | RegexOptions.Singleline);
        markdown = Regex.Replace(markdown, @"<h5[^>]*>(.*?)</h5>", "##### $1\n\n", RegexOptions.IgnoreCase | RegexOptions.Singleline);
        markdown = Regex.Replace(markdown, @"<h6[^>]*>(.*?)</h6>", "###### $1\n\n", RegexOptions.IgnoreCase | RegexOptions.Singleline);

        // Convert paragraphs
        markdown = Regex.Replace(markdown, @"<p[^>]*>(.*?)</p>", "$1\n\n", RegexOptions.IgnoreCase | RegexOptions.Singleline);

        // Convert formatting
        markdown = Regex.Replace(markdown, @"<(strong|b)[^>]*>(.*?)</\1>", "**$2**", RegexOptions.IgnoreCase | RegexOptions.Singleline);
        markdown = Regex.Replace(markdown, @"<(em|i)[^>]*>(.*?)</\1>", "*$2*", RegexOptions.IgnoreCase | RegexOptions.Singleline);
        markdown = Regex.Replace(markdown, @"<code[^>]*>(.*?)</code>", "`$1`", RegexOptions.IgnoreCase | RegexOptions.Singleline);

        // Convert line breaks
        markdown = Regex.Replace(markdown, @"<br[^>]*>", "\n", RegexOptions.IgnoreCase);

        // Convert links
        markdown = Regex.Replace(markdown, @"<a[^>]*href\s*=\s*[""']([^""']*)[""'][^>]*>(.*?)</a>", "[$2]($1)", RegexOptions.IgnoreCase | RegexOptions.Singleline);

        // Convert images
        markdown = Regex.Replace(markdown, @"<img[^>]*src\s*=\s*[""']([^""']*)[""'][^>]*alt\s*=\s*[""']([^""']*)[""'][^>]*>", "![$2]($1)", RegexOptions.IgnoreCase);
        markdown = Regex.Replace(markdown, @"<img[^>]*alt\s*=\s*[""']([^""']*)[""'][^>]*src\s*=\s*[""']([^""']*)[""'][^>]*>", "![$1]($2)", RegexOptions.IgnoreCase);

        // Remove remaining HTML tags
        markdown = Regex.Replace(markdown, @"<[^>]+>", "", RegexOptions.IgnoreCase);

        // Clean up extra whitespace
        markdown = Regex.Replace(markdown, @"\n\s*\n\s*\n", "\n\n");
        markdown = Regex.Replace(markdown, @"^\s+|\s+$", "");

        return markdown;
    }

    private string CleanText(string text)
    {
        // Decode HTML entities
        text = text.Replace("&amp;", "&")
                  .Replace("&lt;", "<")
                  .Replace("&gt;", ">")
                  .Replace("&quot;", "\"")
                  .Replace("&apos;", "'")
                  .Replace("&#39;", "'")
                  .Replace("&nbsp;", " ");

        return text;
    }
}
