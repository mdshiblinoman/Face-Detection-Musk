import { useState } from "react";
import {
  ActivityIndicator,
  Image,
  Pressable,
  SafeAreaView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { StatusBar } from "expo-status-bar";
import { Ionicons } from "@expo/vector-icons";

const API_URL = "http://192.168.0.100:8000";

export default function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function chooseImage(source) {
    setError("");
    setResult(null);
    const options = { mediaTypes: ["images"], quality: 0.85 };
    const response =
      source === "camera"
        ? await ImagePicker.launchCameraAsync(options)
        : await ImagePicker.launchImageLibraryAsync(options);
    if (!response.canceled) {
      setImage(response.assets[0]);
    }
  }

  async function analyzeImage() {
    if (!image) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const body = new FormData();
      body.append("file", {
        uri: image.uri,
        name: "photo.jpg",
        type: image.mimeType || "image/jpeg",
      });
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Prediction failed.");
      setResult(data);
    } catch (requestError) {
      setError(
        `${requestError.message} Make sure the API is running and API_URL uses your PC's local IP.`
      );
    } finally {
      setLoading(false);
    }
  }

  const hasMask = result?.label === "with_mask";

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <View style={styles.container}>
        <View style={styles.header}>
          <View style={styles.logo}>
            <Ionicons name="shield-checkmark" size={25} color="#0B1220" />
          </View>
          <View>
            <Text style={styles.eyebrow}>AI VISION</Text>
            <Text style={styles.title}>Face Mask Check</Text>
          </View>
        </View>

        <Text style={styles.subtitle}>
          Upload a face photo and let the trained model check the mask.
        </Text>

        <View style={styles.previewCard}>
          {image ? (
            <Image source={{ uri: image.uri }} style={styles.preview} />
          ) : (
            <>
              <View style={styles.uploadIcon}>
                <Ionicons name="image-outline" size={38} color="#60A5FA" />
              </View>
              <Text style={styles.emptyTitle}>No photo selected</Text>
              <Text style={styles.emptyText}>Choose from gallery or take a photo</Text>
            </>
          )}
        </View>

        <View style={styles.actions}>
          <Pressable style={styles.secondaryButton} onPress={() => chooseImage("gallery")}>
            <Ionicons name="images-outline" size={20} color="#BFDBFE" />
            <Text style={styles.secondaryText}>Gallery</Text>
          </Pressable>
          <Pressable style={styles.secondaryButton} onPress={() => chooseImage("camera")}>
            <Ionicons name="camera-outline" size={20} color="#BFDBFE" />
            <Text style={styles.secondaryText}>Camera</Text>
          </Pressable>
        </View>

        <Pressable
          style={[styles.analyzeButton, (!image || loading) && styles.disabled]}
          onPress={analyzeImage}
          disabled={!image || loading}
        >
          {loading ? (
            <ActivityIndicator color="#0B1220" />
          ) : (
            <>
              <Text style={styles.analyzeText}>Analyze Photo</Text>
              <Ionicons name="arrow-forward" size={20} color="#0B1220" />
            </>
          )}
        </Pressable>

        {error ? <Text style={styles.error}>{error}</Text> : null}
        {result ? (
          <View style={[styles.result, hasMask ? styles.safeResult : styles.warningResult]}>
            <Ionicons
              name={hasMask ? "checkmark-circle" : "alert-circle"}
              size={28}
              color={hasMask ? "#34D399" : "#FBBF24"}
            />
            <View style={styles.resultCopy}>
              <Text style={styles.resultTitle}>
                {hasMask ? "Mask detected" : "No mask detected"}
              </Text>
              <Text style={styles.resultConfidence}>
                Confidence {(result.confidence * 100).toFixed(1)}%
              </Text>
            </View>
          </View>
        ) : null}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: "#0B1220" },
  container: { flex: 1, padding: 24 },
  header: { flexDirection: "row", alignItems: "center", gap: 14, marginTop: 16 },
  logo: {
    width: 48,
    height: 48,
    borderRadius: 16,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#60A5FA",
  },
  eyebrow: { color: "#60A5FA", fontSize: 11, fontWeight: "800", letterSpacing: 2 },
  title: { color: "#F8FAFC", fontSize: 25, fontWeight: "800", marginTop: 2 },
  subtitle: { color: "#94A3B8", fontSize: 15, lineHeight: 22, marginTop: 22 },
  previewCard: {
    height: 310,
    marginTop: 24,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: "#26354D",
    backgroundColor: "#111C2E",
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
  },
  preview: { width: "100%", height: "100%", resizeMode: "cover" },
  uploadIcon: {
    width: 76,
    height: 76,
    borderRadius: 24,
    backgroundColor: "#172A45",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 16,
  },
  emptyTitle: { color: "#E2E8F0", fontSize: 17, fontWeight: "700" },
  emptyText: { color: "#64748B", fontSize: 13, marginTop: 7 },
  actions: { flexDirection: "row", gap: 12, marginTop: 16 },
  secondaryButton: {
    flex: 1,
    height: 52,
    borderRadius: 15,
    borderWidth: 1,
    borderColor: "#334866",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 9,
  },
  secondaryText: { color: "#BFDBFE", fontSize: 15, fontWeight: "700" },
  analyzeButton: {
    height: 56,
    borderRadius: 16,
    backgroundColor: "#60A5FA",
    marginTop: 16,
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 10,
  },
  disabled: { opacity: 0.4 },
  analyzeText: { color: "#0B1220", fontSize: 16, fontWeight: "800" },
  error: { color: "#FCA5A5", fontSize: 13, lineHeight: 19, marginTop: 16 },
  result: {
    marginTop: 18,
    borderRadius: 18,
    padding: 16,
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  safeResult: { backgroundColor: "#0D302D" },
  warningResult: { backgroundColor: "#3A2D12" },
  resultCopy: { gap: 4 },
  resultTitle: { color: "#F8FAFC", fontSize: 17, fontWeight: "800" },
  resultConfidence: { color: "#CBD5E1", fontSize: 13 },
});
