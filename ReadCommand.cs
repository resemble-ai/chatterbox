using System.CommandLine;
using CSnakes.Runtime;


public class ReadCommand : Command
{
    private readonly Argument<string> sourceOption = new("--source", "Web address, local file or literal plain text to read from.");

    private readonly Option<string?> skipToOption = new("--skipto", "Passage of text to skip to (includes the text, example --skipto \"chapter 2\" will skip all text until case insenstive match is made on that text and read from there).");

    private readonly Option<string?> untilOption = new("--until", "Passage of text to read until to (exlucsive example --until \"chapter 3\").");

    private readonly Option<FileInfo> outputOption = new("--output", () => new FileInfo(Path.Combine(Environment.CurrentDirectory, "output.mp3")), "Output file to write to.");

    private readonly Option<FileInfo> voiceOption = new("--voice", () => new FileInfo(Path.Combine(Environment.CurrentDirectory, "karen.mp3")), "Audio recording of voice to emulate when reading.");

    private readonly Option<double> exagerationOption = new("--exaggeration", () => 0.6, "Exagerration factor 0.5 neutral, 0.8 very exaggerated / dramatic speech.");

    private readonly Option<double> temperaturenOption = new("--temperature", () => 0.8, "Temperature [0.0-1.0] is how much creatitivity or randomness ot introduce (0.8 is the default).");

    private readonly Option<double> cfgOption = new("--cfg", () => 0.5, "CFG [0.0-1.0] is the context-free guidance score, should be roughly inverse to exaggeration, lowering this allows you to compensate when exaggeration speeds up speeach.");
    private readonly IPythonEnvironment environment;

    public ReadCommand(IPythonEnvironment environment) : base("read", "Reads out a web page or a text file")
    {
        this.environment = environment;
        this.Add(sourceOption);
        this.Add(skipToOption);
        this.Add(untilOption);
        this.Add(outputOption);
        this.Add(voiceOption);
        this.Add(exagerationOption);
        this.Add(temperaturenOption);
        this.Add(cfgOption);
        this.SetHandler(this.ReadContentAsync, this.sourceOption, this.skipToOption, this.untilOption, this.outputOption, this.voiceOption, this.exagerationOption, this.temperaturenOption, this.cfgOption);
    }

    public async Task ReadContentAsync(string source, string? skipTo, string? until, FileInfo output, FileInfo conditionalVoice, double exaggeration, double temperature, double cfg)
    {
        var module = this.environment.GenerateWithVoice();
        var seed = 0;
        var readingMode = new ReadingModeExtractor();
        string content = await readingMode.ReadFromSourceAsync(source);
        var filesProduced = new List<string>();

        foreach (var (index, paragraph) in content.Sentences().Paragraphs().SkipWhile(s => !string.IsNullOrWhiteSpace(skipTo) && !s.Contains(skipTo, StringComparison.Ordinal)).TakeWhile(s => string.IsNullOrWhiteSpace(until) || !s.Contains(until, StringComparison.Ordinal)).Index())
        {
            var outputFileName =  $"{Path.GetFileNameWithoutExtension(output.Name)}-{index:000000}.wav";
            var outputPath = Path.Combine(output.Directory!.FullName, outputFileName);
            Console.WriteLine($"{outputFileName}: {paragraph}");

            if (!File.Exists(outputPath))
            {
                module.GenerateAndSave(
                    paragraph,
                    conditionalVoice.FullName,
                    outputPath,
                    exaggeration,
                    temperature,
                    seed,
                    cfg);
            }
            
            filesProduced.Add(outputPath);
        }

        if (filesProduced.Any())
        {
            module.ConcatenateWavFilesSafe(filesProduced, output.FullName);
        }
        else
        {
            Console.WriteLine("Sorry no files produced.");
        }
    }
}