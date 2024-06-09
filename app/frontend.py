import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

# Import your function from the separate file
from functions import predict_disease

def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
    # Display the image
        img = Image.open(file_path)
        img = img.resize((150, 150), Image.LANCZOS) # Resize the image
        img = ImageTk.PhotoImage(img)
        lbl_img.configure(image=img)
        lbl_img.image = img

        # Predict the disease
        disease = predict_disease(file_path)
        print(disease)
        lbl_disease['text'] = "Predicted Disease: " + disease

root = tk.Tk()
root.title("Disease Predictor")

mainframe = ttk.Frame(root, padding="10")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Button to open the file dialog
btn_open = ttk.Button(mainframe, text="Open Image", command=open_file)
btn_open.grid(column=0, row=0, sticky=tk.W)

# Label to display the image
lbl_img = ttk.Label(mainframe)
lbl_img.grid(column=0, row=1, sticky=(tk.W, tk.E, tk.N, tk.S))

# Label to display the predicted disease
lbl_disease = ttk.Label(mainframe, text="Predicted Disease: ")
lbl_disease.grid(column=0, row=2, sticky=tk.W)

root.mainloop()