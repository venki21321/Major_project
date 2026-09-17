package com.example.deepfakedetector;

import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public class MainActivity extends AppCompatActivity {
    private Uri selectedImage;
    private EditText serverUrl;
    private TextView imageName;
    private TextView imageResult;
    private EditText newsText;
    private TextView newsResult;
    private ImageView imagePreview;
    private View uploadPrompt;
    private ProgressBar imageProgress;
    private ProgressBar newsProgress;
    private Button checkImage;
    private Button checkNews;

    private final ActivityResultLauncher<String> imagePicker =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                selectedImage = uri;
                imageName.setText(uri == null ? "No image selected" : displayName(uri));
                if (uri != null) {
                    imagePreview.setImageURI(uri);
                    imagePreview.setVisibility(View.VISIBLE);
                    uploadPrompt.setVisibility(View.GONE);
                    checkImage.setEnabled(true);
                    imageResult.setVisibility(View.GONE);
                }
            });

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.activity_main);
        serverUrl = findViewById(R.id.serverUrl);
        imageName = findViewById(R.id.imageName);
        imageResult = findViewById(R.id.imageResult);
        newsText = findViewById(R.id.newsText);
        newsResult = findViewById(R.id.newsResult);
        imagePreview = findViewById(R.id.imagePreview);
        uploadPrompt = findViewById(R.id.uploadPrompt);
        imageProgress = findViewById(R.id.imageProgress);
        newsProgress = findViewById(R.id.newsProgress);
        checkImage = findViewById(R.id.checkImage);
        checkNews = findViewById(R.id.checkNews);

        serverUrl.setText(getPreferences(MODE_PRIVATE).getString(
                "server_url", "http://10.0.2.2:5001"));
        findViewById(R.id.selectImage).setOnClickListener(v -> imagePicker.launch("image/*"));
        checkImage.setOnClickListener(v -> uploadImage());
        checkNews.setOnClickListener(v -> uploadNews());
    }

    private String baseUrl() {
        String value = serverUrl.getText().toString().trim().replaceAll("/+$", "");
        getPreferences(MODE_PRIVATE).edit().putString("server_url", value).apply();
        return value;
    }

    private void uploadImage() {
        if (selectedImage == null) {
            showResult(imageResult, "Select an image first.", false);
            return;
        }
        if (baseUrl().isEmpty()) {
            showResult(imageResult, "Enter the backend server URL.", false);
            return;
        }
        setLoading(true, true);
        new Thread(() -> {
            try {
                String boundary = "----Deepfake" + System.currentTimeMillis();
                HttpURLConnection connection = open("/api/predict-image", "POST");
                connection.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);
                try (OutputStream out = connection.getOutputStream()) {
                    String header = "--" + boundary
                            + "\r\nContent-Disposition: form-data; name=\"file\"; filename=\""
                            + displayName(selectedImage)
                            + "\"\r\nContent-Type: image/jpeg\r\n\r\n";
                    out.write(header.getBytes(StandardCharsets.UTF_8));
                    try (InputStream in = getContentResolver().openInputStream(selectedImage)) {
                        if (in == null) throw new IllegalStateException("Cannot read the selected image.");
                        byte[] buffer = new byte[8192];
                        int count;
                        while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
                    }
                    out.write(("\r\n--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
                }
                JSONObject json = response(connection);
                showResult(imageResult,
                        json.optString("prediction", json.optString("error")),
                        json.has("prediction"));
            } catch (Exception error) {
                showResult(imageResult, "Connection failed: " + friendlyMessage(error), false);
            } finally {
                runOnUiThread(() -> setLoading(true, false));
            }
        }).start();
    }

    private void uploadNews() {
        String text = newsText.getText().toString().trim();
        if (text.length() < 20) {
            showResult(newsResult, "Enter at least 20 characters for a useful analysis.", false);
            return;
        }
        if (baseUrl().isEmpty()) {
            showResult(newsResult, "Enter the backend server URL.", false);
            return;
        }
        setLoading(false, true);
        new Thread(() -> {
            try {
                HttpURLConnection connection = open("/api/predict-news", "POST");
                connection.setRequestProperty("Content-Type", "application/json");
                byte[] body = new JSONObject().put("text", text).toString()
                        .getBytes(StandardCharsets.UTF_8);
                try (OutputStream out = connection.getOutputStream()) {
                    out.write(body);
                }
                JSONObject json = response(connection);
                showResult(newsResult,
                        json.optString("prediction", json.optString("error")),
                        json.has("prediction"));
            } catch (Exception error) {
                showResult(newsResult, "Connection failed: " + friendlyMessage(error), false);
            } finally {
                runOnUiThread(() -> setLoading(false, false));
            }
        }).start();
    }

    private HttpURLConnection open(String path, String method) throws Exception {
        HttpURLConnection connection = (HttpURLConnection) new URL(baseUrl() + path).openConnection();
        connection.setRequestMethod(method);
        connection.setDoOutput(true);
        connection.setConnectTimeout(15000);
        connection.setReadTimeout(120000);
        return connection;
    }

    private JSONObject response(HttpURLConnection connection) throws Exception {
        int status = connection.getResponseCode();
        InputStream stream = status < 400
                ? connection.getInputStream() : connection.getErrorStream();
        if (stream == null) throw new IllegalStateException("Server returned HTTP " + status);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        byte[] buffer = new byte[4096];
        int count;
        while ((count = stream.read(buffer)) != -1) bytes.write(buffer, 0, count);
        stream.close();
        connection.disconnect();
        return new JSONObject(bytes.toString(StandardCharsets.UTF_8));
    }

    private String displayName(Uri uri) {
        try (android.database.Cursor cursor = getContentResolver().query(
                uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) return cursor.getString(index);
            }
        }
        return "image.jpg";
    }

    private void setLoading(boolean image, boolean loading) {
        ProgressBar progress = image ? imageProgress : newsProgress;
        Button button = image ? checkImage : checkNews;
        progress.setVisibility(loading ? View.VISIBLE : View.GONE);
        button.setEnabled(!loading && (!image || selectedImage != null));
        button.setText(loading ? "Analyzing…" : image ? "Analyze image" : "Analyze text");
    }

    private void showResult(TextView target, String text, boolean success) {
        runOnUiThread(() -> {
            target.setText(text == null || text.isEmpty() ? "No result returned." : text);
            target.setTextColor(getColor(success ? R.color.ink : android.R.color.holo_red_dark));
            target.setVisibility(View.VISIBLE);
        });
    }

    private String friendlyMessage(Exception error) {
        String message = error.getMessage();
        if (message == null || message.trim().isEmpty()) return "Unable to reach the server.";
        String lower = message.toLowerCase();
        if (lower.contains("failed to connect") || lower.contains("connection refused")) {
            return "Backend unavailable. Check the server URL and Wi-Fi connection.";
        }
        return message;
    }
}
