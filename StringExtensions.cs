using System.Text.RegularExpressions;
using CommunityToolkit.HighPerformance.Buffers;

public static class StringExtensions
{
    private static readonly string[] ForceSplitParagraphs = ["#", "CHAPTER", "INTRODUCTION", "CONTENTS", "**", "* ", "[", "---"];
    private static readonly string[] NoSplitParagraphs = ["mrs.", "mr."];
    private static readonly string[] PunctuationSuffixes = [". ", "! ", "; ", "? ", ".\""];

    private static readonly (Regex, string)[] pronunciationFixes =
    [
        (new Regex("\\bass\\b", RegexOptions.IgnoreCase), "arse"),
        //(new Regex("\\bnorthanger\\b", RegexOptions.IgnoreCase), "Nor thanger"),
        (new Regex("\\bworth-while\\b", RegexOptions.IgnoreCase), "worthwhile"),
        (new Regex("\\babbey\\b", RegexOptions.IgnoreCase), "abbey"),
    ];

    public static IEnumerable<string> Sentences(this string content)
    {
        content = content.Replace("\n", " ")
                    .Replace(new string([' ', (char)65533]), string.Empty)
                    .Replace(new string([(char)65533]), string.Empty)
                    .Replace(new string([' ', (char)8260]), string.Empty)
                    .Replace("”", "\"")
                    .Replace("“", "\"");

        content = pronunciationFixes.Aggregate(content, (p, fix) => fix.Item1.Replace(p, fix.Item2));
        var sentences = content.SplitBeforeAny(ForceSplitParagraphs)
                .SelectMany(s => s.SplitAfterAny(PunctuationSuffixes))
                .Where(s => !string.IsNullOrWhiteSpace(s) && !(s.Trim().StartsWith("[") && s.Trim().EndsWith(")")))
                .ToList();
        return sentences;
    }

    public static IEnumerable<string> Paragraphs(this IEnumerable<string> sentences, int maxParagraphWordCount = 40)
    {
        var current = new List<string>();
        foreach (var sentence in sentences)
        {
            var wordCount = current.Count > 0 ? current.Sum(s => s.Count(c => c == ' ')) : 0;
            var lastSentence = current.LastOrDefault() ?? string.Empty;
            if ((ForceSplitParagraphs.Any(s => sentence.StartsWith(s, StringComparison.OrdinalIgnoreCase)) || wordCount > maxParagraphWordCount) && !NoSplitParagraphs.Any(s => lastSentence.EndsWith(s, StringComparison.OrdinalIgnoreCase)))
            {
                yield return string.Concat(current).Trim();
                current = new List<string>();
            }

            current.Add(sentence);
        }

        if (current.Count > 0)
        {
            yield return string.Concat(current).Trim();
        }
    }

    public static IEnumerable<string> SplitBeforeAny(this string text, params string[] splitters)
    {
        if (string.IsNullOrEmpty(text) || splitters == null || splitters.Length == 0)
        {
            yield return text ?? string.Empty;
            yield break;
        }

        int currentIndex = 0;

        while (currentIndex < text.Length)
        {
            int nearestSplitterIndex = -1;
            string? nearestSplitter = null;

            // Find the nearest splitter
            foreach (string splitter in splitters)
            {
                if (string.IsNullOrEmpty(splitter)) continue;

                int index = text.IndexOf(splitter, currentIndex, StringComparison.Ordinal);
                if (index != -1 && (nearestSplitterIndex == -1 || index < nearestSplitterIndex))
                {
                    nearestSplitterIndex = index;
                    nearestSplitter = splitter;
                }
            }

            if (nearestSplitterIndex == -1)
            {
                // No more splitters found, return the rest of the string
                if (currentIndex < text.Length)
                {
                    yield return text.Substring(currentIndex);
                }
                break;
            }
            else
            {
                // Return text up to (but not including) the splitter
                if (nearestSplitterIndex > currentIndex)
                {
                    yield return text.Substring(currentIndex, nearestSplitterIndex - currentIndex);
                }

                // Move past the splitter for next iteration
                currentIndex = nearestSplitterIndex + (nearestSplitter?.Length ?? 0);
            }
        }
    }

    public static IEnumerable<string> SplitAfterAny(this string text, params string[] splitters)
    {
        //  Split a string (without consuming the text but yielding the string to that point)
        if (string.IsNullOrEmpty(text))
        {
            yield break;
        }

        if (splitters.Length == 0)
        {
            yield return text;
            yield break;
        }

        int currentIndex = 0;
        while (currentIndex < text.Length)
        {
            int nextSplitIndex = -1;
            string? foundSplitter = null;

            // Find the earliest occurrence of any splitter
            foreach (string splitter in splitters)
            {
                if (string.IsNullOrEmpty(splitter))
                    continue;

                int index = text.IndexOf(splitter, currentIndex, StringComparison.Ordinal);
                if (index != -1 && (nextSplitIndex == -1 || index < nextSplitIndex))
                {
                    nextSplitIndex = index;
                    foundSplitter = splitter;
                }
            }

            if (nextSplitIndex == -1)
            {
                // No more splitters found, return the rest of the string
                if (currentIndex < text.Length)
                {
                    yield return text.Substring(currentIndex);
                }

                break;
            }
            else
            {
                // Return the substring including the splitter
                int endIndex = nextSplitIndex + foundSplitter?.Length ?? 0;
                yield return text.Substring(currentIndex, endIndex - currentIndex);
                currentIndex = endIndex;
            }
        }
    }
}