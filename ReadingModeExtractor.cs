using System.Text;
using System.Text.RegularExpressions;
using AngleSharp;
using AngleSharp.Dom;
using AngleSharp.Html.Dom;

public class ReadingModeExtractor
{
	private readonly IConfiguration _config;
	private readonly IBrowsingContext _context;

	// Elements to completely remove
	private readonly HashSet<string> _excludedTags = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
	{
		"script", "style", "nav", "header", "footer", "aside", "form",
		"button", "input", "select", "textarea", "iframe", "object",
		"embed", "canvas", "svg", "audio", "video", "noscript", "comment"
	};

	// Block-level elements that should have line breaks
	private readonly HashSet<string> _blockElements = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
	{
		"div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "article", "section",
		"blockquote", "pre", "ul", "ol", "li", "table", "tr", "td", "th",
		"br", "hr", "main", "address"
	};

	// Attributes that might indicate non-content elements
	private readonly HashSet<string> _excludedClasses = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
	{
		"advertisement", "ads", "ad", "sidebar", "menu", "navigation",
		"footer", "header", "social", "share", "comment", "popup",
		"modal", "overlay", "banner", "toolbar", "breadcrumb"
	};

	public ReadingModeExtractor()
	{
		_config = Configuration.Default;
		_context = BrowsingContext.New(_config);
	}

    public async Task<string> ExtractReadingText(string html)
    {
        var document = await _context.OpenAsync(req => req.Content(html));
        if (document is IHtmlDocument doc)
        {
            return ExtractReadingText(doc);
        }

        return string.Empty;
	}

	public async Task<string> ExtractReadingTextFromUrl(string url)
	{
		var text = await new HttpClient().GetStringAsync(url);
		return await ExtractReadingText(text);
	}

	public string ExtractReadingText(IHtmlDocument document)
	{
		// Remove unwanted elements
		RemoveUnwantedElements(document);

		// Find the main content area
		var mainContent = FindMainContent(document);

		// Extract text with proper formatting
		var text = ExtractFormattedText(mainContent);

		// Clean up the text
		return CleanText(text);
	}

	private void RemoveUnwantedElements(IHtmlDocument document)
	{
		// Remove script, style, and other unwanted tags
		foreach (var tag in _excludedTags)
		{
			var elements = document.QuerySelectorAll(tag)?.ToList() ?? [];
			foreach (var element in elements)
			{
				element.Remove();
			}
		}

		// Remove elements with excluded class names
		var elementsWithClasses = document.QuerySelectorAll("[class]")?.ToList() ?? [];
		foreach (var element in elementsWithClasses)
		{
			var classes = element.GetAttribute("class")?.ToLower().Split(' ') ?? Array.Empty<string>();
			if (classes.Any(c => _excludedClasses.Contains(c)))
			{
				element.Remove();
			}
		}

		// Remove hidden elements
		var hiddenElements = document.QuerySelectorAll("[style*='display: none'], [style*='visibility: hidden'], [hidden]").ToList();
		foreach (var element in hiddenElements)
		{
			element.Remove();
		}
	}

	private IElement FindMainContent(IHtmlDocument document)
	{
		// Try to find main content using semantic elements and common patterns
		var candidates = new List<IElement>();

		// Look for semantic main content
		var main = document.QuerySelector("main");
		if (main != null) candidates.Add(main);

		var article = document.QuerySelector("article");
		if (article != null) candidates.Add(article);

		// Look for content-indicating class names or IDs
		var contentSelectors = new[]
		{
			"[id*='content']", "[class*='content']", "[id*='main']", "[class*='main']",
			"[id*='article']", "[class*='article']", "[id*='post']", "[class*='post']",
			"[class*='entry']", "[id*='body']", "[class*='body']"
		};

		foreach (var selector in contentSelectors)
		{
			var elements = document.QuerySelectorAll(selector);
			candidates.AddRange(elements);
		}

		// Score candidates based on text content density
		IElement? bestCandidate = null;
		int bestScore = 0;

		foreach (var candidate in candidates.Distinct())
		{
			var score = ScoreElement(candidate);
			if (score > bestScore)
			{
				bestScore = score;
				bestCandidate = candidate;
			}
		}

		// Fall back to body if no good candidate found
		return bestCandidate ?? document.Body ?? document.DocumentElement;
	}

	private int ScoreElement(IElement element)
	{
		if (element == null) return 0;

		var text = element.TextContent;
		if (string.IsNullOrWhiteSpace(text)) return 0;

		var score = 0;

		// Base score on text length
		score += Math.Min(text.Length / 100, 100);

		// Bonus for paragraphs
		score += element.QuerySelectorAll("p").Length * 10;

		// Bonus for headers
		score += element.QuerySelectorAll("h1, h2, h3, h4, h5, h6").Length * 5;

		// Penalty for excessive links (might be navigation)
		var linkRatio = (double)element.QuerySelectorAll("a").Length / Math.Max(text.Split(' ').Length, 1);
		if (linkRatio > 0.3) score -= 50;

		// Penalty for very short text blocks
		if (text.Length < 200) score -= 20;

		return score;
	}

	private string ExtractFormattedText(IElement element)
	{
		var sb = new StringBuilder();
		ExtractTextRecursive(element, sb);
		return sb.ToString();
	}

	private void ExtractTextRecursive(INode node, StringBuilder sb)
	{
		if (node is IText textNode)
		{
			var text = textNode.TextContent?.Trim();
			if (!string.IsNullOrEmpty(text))
			{
				sb.Append(text);
				sb.Append(" ");
			}
		}
		else if (node is IElement element)
		{
			var tagName = element.TagName.ToLower();

			// Skip excluded elements
			if (_excludedTags.Contains(tagName))
				return;

			// Add line breaks for block elements
			if (_blockElements.Contains(tagName))
			{
				sb.AppendLine();
			}

			// Special handling for specific elements
			switch (tagName)
			{
				case "h1":
				case "h2":
				case "h3":
				case "h4":
				case "h5":
				case "h6":
					sb.AppendLine();
					sb.AppendLine("### " + element.TextContent?.Trim());
					sb.AppendLine();
					return;

				case "li":
					sb.Append("• ");
					break;

				case "blockquote":
					sb.AppendLine();
					sb.Append("> ");
					break;

				case "br":
					sb.AppendLine();
					return;

				case "hr":
					sb.AppendLine();
					sb.AppendLine("---");
					sb.AppendLine();
					return;
			}

			// Process child nodes
			foreach (var child in element.ChildNodes)
			{
				ExtractTextRecursive(child, sb);
			}

			// Add line breaks after block elements
			if (_blockElements.Contains(tagName))
			{
				sb.AppendLine();
			}
		}
	}

	private string CleanText(string text)
	{
		if (string.IsNullOrEmpty(text))
			return string.Empty;

		// First, normalize line endings
		text = text.Replace("\r\n", "\n").Replace("\r", "\n");

		// Remove excessive consecutive spaces within lines (but preserve single spaces)
		text = Regex.Replace(text, @"[ \t]+", " ");

		// Remove excessive line breaks (3+ consecutive newlines become 2)
		text = Regex.Replace(text, @"\n{3,}", "\n\n");

		// Clean up empty list bullets
		text = Regex.Replace(text, @"^\s*•\s*$", "", RegexOptions.Multiline);

		// Remove leading/trailing whitespace from each line, but preserve the line structure
		var lines = text.Split('\n')
			.Select(line => line.Trim())
			.ToList();

		// Remove completely empty lines only if they create excessive spacing
		var cleanedLines = new List<string>();
		bool lastWasEmpty = false;

		foreach (var line in lines)
		{
			if (string.IsNullOrEmpty(line))
			{
				if (!lastWasEmpty) // Allow one empty line for paragraph breaks
				{
					cleanedLines.Add(line);
					lastWasEmpty = true;
				}
			}
			else
			{
				cleanedLines.Add(line);
				lastWasEmpty = false;
			}
		}

		return string.Join("\n", cleanedLines).Trim();
	}

    public async Task<string> ReadFromSourceAsync(string source)
    {
        if (source.StartsWith("http"))
        {
            return await this.ExtractReadingTextFromUrl(source);
        }

        if (File.Exists(source))
        {
            var extension = Path.GetExtension(source);
            if (extension is ".html" or ".htm")
            {
                return await this.ExtractReadingText(await File.ReadAllTextAsync(source));
            }
            else if (extension is ".epub")
            {
                var readingMode = new EpubToMarkdownConverter();
                return await readingMode.ConvertEpubToMarkdown(source);
            }

            // Assume some kind of .txt file.
            return await File.ReadAllTextAsync(source);
        }

        return source;
    }
}
