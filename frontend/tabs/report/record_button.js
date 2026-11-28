// Setup if needed and start recording.
async () => {
  // Set up recording functions if not already initialized
  if (!window.startRecording) {
      let recorder_js = null;
      let main_js = null;
  }

  // Function to fetch and convert video blob to base64 using async/await without explicit Promise
  async function getVideoBlobAsBase64(objectURL) {
      const response = await fetch(objectURL);
      if (!response.ok) {
        throw new Error('Failed to fetch video blob.');
      }

      const blob = await response.blob();

      const reader = new FileReader();
      reader.readAsDataURL(blob);

      return new Promise((resolve, reject) => {
        reader.onloadend = () => {
          if (reader.result) {
            resolve(reader.result.split(',')[1]); // Return the base64 string (without data URI prefix)
          } else {
            reject('Failed to convert blob to base64.');
          }
        };
      });
  }

  if (window.currentState === "RECORDING") {
      await window.stopRecording();
      const base64String = await getVideoBlobAsBase64(window.videoSource);
      return base64String;
  } else {
      window.startRecording();
      return "Record";
  }
}
