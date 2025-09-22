import re
import os


def sanitize_filename(title):
    """
    Cleans a story title to create a valid filename.
    Removes Roman numerals, converts to lowercase, and replaces spaces.
    """
    # Remove Roman numeral and period (e.g., "I. ", "XII. ")
    name = re.sub(r"^[IVXLC]+\.\s+", "", title).strip()
    # Convert to lowercase and replace spaces with underscores
    name = name.lower().replace(" ", "_")
    # Remove any characters that are not letters, numbers, or underscores
    name = re.sub(r"[^a-z0-9_]", "", name)
    return name


def preprocess_and_split_stories(input_file, output_dir):
    """
    Reads the full text, removes the header/footer, and splits it
    into individual files for each short story.
    """
    # 1. Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured output directory exists: '{output_dir}'")

    # 2. Read the entire book content
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            full_text = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return

    # 3. Pre-process: Isolate the main content of the stories
    try:
        start_marker = "I. A SCANDAL IN BOHEMIA"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

        start_index = full_text.find(start_marker)
        end_index = full_text.find(end_marker)

        if start_index == -1 or end_index == -1:
            raise ValueError(
                "Could not find start or end markers for the book content."
            )

        # Extract the core text containing all the stories
        core_text = full_text[start_index:end_index].strip()
        print("Successfully isolated the core story content.")

    except ValueError as e:
        print(f"Error during pre-processing: {e}")
        return

    # 4. Split the core text by story
    # The pattern matches lines that are Roman numerals, a dot, and an all-caps title.
    split_pattern = r"^[IVXLC]+\.\s+[A-Z\s]+$"

    # Use a capturing group `()` to keep the titles in the resulting list
    # The `re.MULTILINE` flag allows `^` to match the start of each line
    parts = re.split(f"({split_pattern})", core_text, flags=re.MULTILINE)

    if len(parts) < 3:
        print(
            "Error: Failed to split the text into stories. Check the delimiter pattern."
        )
        return

    # Group the list into (title, content) pairs
    stories = zip(parts[1::2], parts[2::2])

    # 5. Save each story to its own file
    for i, (title, content) in enumerate(stories, 1):
        filename = f"{i:02d}_{sanitize_filename(title)}.txt"
        output_path = os.path.join(output_dir, filename)

        # Combine the title and the story content
        full_story_text = (title.strip() + "\n\n" + content.strip()).strip()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_story_text)

        print(f"Successfully created: {output_path}")

    print(
        f"\n✅ Processing complete. {i} stories have been split into the '{output_dir}' directory."
    )


# --- How to use the script ---
if __name__ == "__main__":
    # The name of your downloaded book file
    book_filepath = r"C:\Learning\standard-rag-text\data\sherlock.txt"

    # The name of the folder where the individual story files will be saved
    stories_output_folder = "sherlock_stories"

    preprocess_and_split_stories(book_filepath, stories_output_folder)
