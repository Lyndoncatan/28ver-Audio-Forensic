# Audio Forensic Detector

Advanced Audio Analysis & Instrument Detection System

## 🚀 Live Demo

Visit the live application: [Audio Forensic Detector on Render](https://your-app-name.onrender.com)

## Features

- 🎵 Real-time audio recording and analysis
- 📁 Audio file upload support (MP3, WAV, M4A, OGG)
- 🔍 Comprehensive forensic audio analysis
- 📊 Interactive visualizations and sonar view
- 📄 PDF report generation
- ⚙️ Configurable analysis settings
- 💾 Cloud database storage (Supabase)

## Technology Stack

- **Frontend**: Next.js 14, React 18, TypeScript
- **Styling**: Tailwind CSS, Radix UI
- **Audio Processing**: Web Audio API, Canvas API
- **Database**: Supabase (PostgreSQL)
- **PDF Generation**: jsPDF
- **Deployment**: Render

## Local Development

1. Clone the repository
2. Install dependencies: `npm install`
3. Set up environment variables in `.env.local`
4. Run development server: `npm run dev`
5. Open [http://localhost:3000](http://localhost:3000)

## Environment Variables

\`\`\`env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
\`\`\`

## Browser Requirements

- Modern browser with Web Audio API support
- HTTPS connection for microphone access
- JavaScript enabled

## Deployment

This project is optimized for Render deployment with:
- Automatic builds from Git
- Environment variable configuration
- Production optimizations
- Static asset handling

## License

MIT License - see LICENSE file for details
