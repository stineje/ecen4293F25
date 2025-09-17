from PyPDF2 import PdfReader, PdfWriter

# open file in binary
with open("first.pdf", "rb") as file:
    reader = PdfReader(file)
    num_pages = len(reader.pages)
    print(f"Number of pages: {num_pages}")
    # Access the first page and rotate it
    page = reader.pages[0]
    page.rotate(90)  # Use the rotate method in PyPDF2 v3.0.0
    
    # Create a writer object and add the rotated page
    writer = PdfWriter()
    writer.add_page(page)
    writer.insert_blank_page()
    
    # Write the rotated page to a new file
    with open("rotated_output.pdf", "wb") as output_file:
        writer.write(output_file)
