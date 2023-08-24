# Set the theme of Gradio.
# You can get more themes at https://huggingface.co/spaces/gradio/theme-gallery
# For example if you want this one: https://huggingface.co/spaces/bethecloud/storj_theme
# You will have to look at a line that starts like "To use this theme, set"
# On the same line look for [" theme='[AUTHOR]/[THEME]' "]. e.g. [" theme='bethecloud/storj_theme' "]
# Copy just the part in apostrophes: ''. e.g. bethecloud/storj_theme
# Now modify the line below and paste that part with replacement in quotation mark. e.g. "bethecloud/storj_theme"
# In the end you should have THEME = "bethecloud/storj_theme"
$props = @{
    PYCMD = "runtime\python.exe"
    PORT = "7897"
    THEME = "gradio/soft" # Modify accordigly to change Gradio theme
}

Write-Host "Current Settings:`n" -ForegroundColor Magenta
# Display the current settings
$props.GetEnumerator() | ForEach-Object { 
    Write-Host ("{0}:" -f $_.Name) -NoNewline -ForegroundColor Green
    Write-Host (" {0}" -f $_.Value) -ForegroundColor Cyan
}

Write-Host ""

# Run Python script using properties as arguments
& $props.PYCMD infer-web.py --pycmd $props.PYCMD --port $props.PORT --theme $props.THEME

# Pause the script at the end
Write-Host "Press any key to continue ..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")