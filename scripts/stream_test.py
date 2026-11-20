import cv2
import yt_dlp


def get_stream_url(youtube_url: str, max_height: int = 1080) -> str:
    """Extrae la URL directa del manifiesto HLS (.m3u8) evitando el bloqueo de cliente web."""
    ydl_opts = {
        "format": f"best[height<={max_height}]/best",
        "quiet": True,
        "no_warnings": True,
        # Clave para evitar "The page needs to be reloaded": usar clientes tv / ios
        "extractor_args": {
            "youtube": {
                "player_client": ["tv", "ios", "android"],
            }
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


def main(youtube_url: str, max_height: int = 1080):
    print("Extrayendo URL del stream con yt-dlp...")
    try:
        stream_url = get_stream_url(youtube_url, max_height=max_height)
        print("URL extraída correctamente. Abriendo OpenCV...")
    except Exception as e:
        print(f"Error al extraer el stream: {e}")
        return

    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("No se pudo abrir el stream en OpenCV.")
        return

    window_name = "Airport Live Stream - pulsa 'q' para salir"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame perdido o stream desconectado.")
                break

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    LIVE_URL = "https://www.youtube.com/watch?v=4nC8r5neaow"
    main(youtube_url=LIVE_URL, max_height=1080)