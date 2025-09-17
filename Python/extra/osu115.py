from PyPDF2 import PdfReader, PdfWriter

def merge_pdfs(pdf1_path, pdf2_path, output_path):
    # Create a PDF writer object
    writer = PdfWriter()

    # Read the first PDF file
    with open(pdf1_path, 'rb') as pdf1_file:
        reader1 = PdfReader(pdf1_file)
        for page in reader1.pages:
            writer.add_page(page)  # Add all pages from the first PDF

    # Read the second PDF file
    with open(pdf2_path, 'rb') as pdf2_file:
        reader2 = PdfReader(pdf2_file)
        for page in reader2.pages:
            writer.add_page(page)  # Add all pages from the second PDF

    # Write the merged PDF to an output file
    with open(output_path, 'wb') as output_file:
        writer.write(output_file)

    print(f'Merged PDF saved as {output_path}')

# Usage example:
merge_pdfs('first.pdf', 'second.pdf', 'merged_output.pdf')
