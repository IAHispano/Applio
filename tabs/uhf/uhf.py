import os
import gradio as gr
import zipfile

def zip_and_upload(repo_id, file1, file2, zip_name):
    # Create a ZipFile object
    with zipfile.ZipFile(zip_name + '.zip', 'w') as zipf:
        # Write file1 to the zip
        zipf.write(file1, os.path.basename(file1))
        # Write file2 to the zip
        zipf.write(file2, os.path.basename(file2))
    
    api.upload_file(
        path_or_fileobj=f"{zip_name}.zip",
        path_in_repo=f"{zip_name}.zip",
        repo_id=repo_id,
        repo_type="model"
    )


def uhf():
    with gr.Column():

        repo_id = gr.Textbox(label=("repo huggingface id"), info="your repository id")
        file1 = gr.Textbox(label=("pth file path"), info="your pth file")
        file2 = gr.Textbox(label=("index file path"), info="your index file")
        zip_name = gr.Textbox(label=("zip name models"), info="your zip models name")

      
        output_info = gr.Textbox(
            label=("Output Information"),
            info=("The output information will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )
        start_button = gr.Button(
            value=("start upload your file!"), variant="primary"
        
    start_button_button.click(
        fn=zip_and_upload,
        inputs=[repo_id, file1, file2, zip_name],
        outputs=[output_info],
    )
