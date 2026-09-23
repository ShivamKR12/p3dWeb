# p3dWeb - Panda3D on the Web

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Panda3D](https://img.shields.io/badge/Engine-Panda3D-15427b?logo=python&logoColor=white)
![WebAssembly](https://img.shields.io/badge/Target-WebAssembly-654FF0?logo=webassembly&logoColor=white)

</div>

**p3dWeb** is an experimental 3D graphics demonstration showcasing the power of the Panda3D game engine running natively in the browser. By leveraging Python web-porting tools like `pygbag`, this project demonstrates how complex 3D environments, character models, and interactive scenes can be deployed directly to web users without requiring any local installation.

## 🌟 Features

- **Panda3D Rendering Engine**: Utilizes the robust rendering capabilities of Panda3D for 3D environments.
- **WebAssembly Deployment**: Ready for browser execution. The game is packaged and compiled to WebAssembly.
- **Rich Assets**: Includes classic Panda3D demonstration models (like the iconic Panda) and various environment maps (mountains, trees, grass, rocks).
- **Automated Web Builds**: Fully integrated with GitHub Actions (`pygbag.yml`) to automatically compile and deploy the latest web build upon every commit.

## 🛠️ Local Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/p3dWeb.git
   cd p3dWeb
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```
   *Windows:* `.\venv\Scripts\activate`
   *Linux/macOS:* `source venv/bin/activate`

3. **Install dependencies**
   ```bash
   pip install panda3d pygbag
   ```

## 🚀 Running Locally

To run the application natively on your desktop:
```bash
python main.py
```

## 🌐 Browser / Web Setup

This project is explicitly configured for web deployment. To test the web build locally:

1. **Build for the web**
   From the project root, run:
   ```bash
   pygbag main.py
   ```
2. **Play in Browser**
   Navigate to the local server address provided by `pygbag` (usually `http://localhost:8000`).

## 📁 Project Structure

```text
p3dWeb/
├── .github/workflows/       # Automated web build pipeline (pygbag.yml)
├── assets/                  # 2D textures (UI, crosshair, dirt, grass, logos)
├── models/                  # 3D models and environments
│   ├── maps/                # Environment textures (bamboo, rocks, mountains)
│   ├── environment.egg.pz   # Compressed 3D environment geometry
│   ├── panda-model.egg.pz   # The main Panda character model
│   └── panda-walk4.egg.pz   # Panda walking animation data
├── main.py                  # Panda3D application entry point
└── .gitignore               # Ignored files
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
