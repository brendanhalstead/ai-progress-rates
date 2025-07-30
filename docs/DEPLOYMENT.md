# Deployment Guide

Quick deployment options for the AI Progress Modeling Web App.

## 🚀 Railway (Recommended for Hackathons)

1. **Fork the repository** on GitHub
2. **Connect to Railway:**
   - Go to [railway.app](https://railway.app)
   - Click "Start a New Project"
   - Choose "Deploy from GitHub repo"
   - Select your forked repository
3. **Deploy automatically** - Railway will detect the Python app and deploy!
4. **Get your URL** - Railway provides a public URL

**That's it!** Railway automatically:
- Installs dependencies from `requirements.txt`
- Uses the `Procfile` to start with Gunicorn
- Sets the PORT environment variable

## 🚀 Render

1. **Fork the repository** on GitHub
2. **Connect to Render:**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
3. **Configure:**
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn --bind 0.0.0.0:$PORT app:app`
4. **Deploy** - Click "Create Web Service"

## 🚀 Heroku

1. **Fork the repository** on GitHub
2. **Create Heroku app:**
   ```bash
   heroku create your-app-name
   ```
3. **Deploy:**
   ```bash
   git push heroku main
   ```
4. **Open your app:**
   ```bash
   heroku open
   ```

## 🚀 Local Testing

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run with Gunicorn (production-like)
gunicorn --bind 0.0.0.0:5000 app:app

# Or run with Flask dev server
python app.py
```

## 📋 Files Added for Deployment

- `Procfile` - Tells platforms how to start the app
- `requirements.txt` - Updated with `gunicorn` dependency
- Modified `app.py` - Uses PORT environment variable

## 🔧 Environment Variables

For production, you can set:
- `PORT` - Port number (auto-set by most platforms)
- `FLASK_ENV` - Set to "production" for production mode

## 📱 Features

The deployed app includes:
- Interactive parameter sliders
- Real-time visualization updates
- CSV upload/download
- Parameter estimation
- Responsive mobile-friendly design

## 🐛 Troubleshooting

**App won't start?**
- Check logs on your platform's dashboard
- Verify all dependencies in `requirements.txt`

**Parameters not working?**
- Clear browser cache
- Check JavaScript console for errors

**File uploads failing?**
- Ensure CSV format matches expected columns
- Check file size < 16MB

## 🎯 Quick Demo

Once deployed, the app provides:
1. **Adjust sliders** to change AI progress parameters
2. **See real-time updates** in interactive plots
3. **Upload custom data** via CSV files
4. **Export results** for further analysis

Perfect for demonstrating AI progress scenarios at conferences, hackathons, or research presentations!