import qrcode
from PIL import Image

# ---- 1) Your link goes here ----
data = "https://www.okstate.edu"

# ---- 2) Build the QR code (high error correction allows logo) ----
qr = qrcode.QRCode(
    version=2,  # controls density; bump up if logo is large
    error_correction=qrcode.constants.ERROR_CORRECT_H,  # High correction
    box_size=10,
    border=4,
)
qr.add_data(data)
qr.make(fit=True)

# ---- 3) Make QR code into an image ----
qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

# ---- 4) Open your logo (must already be a square or will scale) ----
logo = Image.open("logo.png")

# ---- 5) Resize logo relative to QR size ----
qr_size = qr_img.size[0]
logo_size = qr_size // 4  # logo is 25% of QR width
logo = logo.resize((logo_size, logo_size), Image.LANCZOS)

# ---- 6) Center the logo ----
pos = ((qr_size - logo_size) // 2, (qr_size - logo_size) // 2)
qr_img.paste(logo, pos, mask=logo)  # Make sure logo has transparency!

# ---- 7) Save the final result ----
qr_img.save("qr_with_logo.png")

print("QR code with embedded logo saved to qr_with_logo.png")
