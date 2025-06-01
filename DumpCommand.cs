using System.CommandLine;
using CSnakes.Runtime;
public class DumpCommand : Command
{
    private readonly Argument<string> sourceOption = new("--source",  "Web address, local file or literal plain text to read from.");

    private readonly Option<string?> skipToOption = new("--skipto", "Passage of text to skip to (includes the text, example --skipto \"chapter 2\" will skip all text until case insenstive match is made on that text and read from there).");

    private readonly Option<string?> untilOption = new("--until", "Passage of text to read until to (exlucsive example --until \"chapter 3\").");



    public DumpCommand() : base("dump", "Dumps out each sentence and paragraph.")
    {
        this.Add(this.sourceOption);
        this.Add(this.skipToOption);
        this.Add(this.untilOption);
        this.SetHandler(this.DumpText, this.sourceOption, this.skipToOption, this.untilOption);
    }

    public async Task DumpText(string source, string? skipTo, string? until)
    {
        var readingMode = new ReadingModeExtractor();
        string content = await readingMode.ReadFromSourceAsync(source);
        foreach (var (index, paragraph) in content.Sentences().Paragraphs().SkipWhile(s => !string.IsNullOrWhiteSpace(skipTo) && !s.Contains(skipTo, StringComparison.Ordinal)).TakeWhile(s => string.IsNullOrWhiteSpace(until) || !s.Contains(until, StringComparison.Ordinal)).Index())
        {
            Console.WriteLine($"{index:0000000}: {paragraph}");
            Console.WriteLine();
        }
    }
}