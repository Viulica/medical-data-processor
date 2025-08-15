# PDF Processing Automation Tool

A modern web application for automating the extraction of structured data from patient PDF documents using AI.

## Features

- **Drag & Drop Interface**: Easy file upload for ZIP archives and Excel instructions
- **Configurable Processing**: Set the number of pages to extract per patient PDF
- **Real-time Progress**: Live status updates during processing
- **Batch Processing**: Process multiple patient PDFs simultaneously
- **CSV Export**: Download results in a structured format
- **Modern UI**: Built with Vue.js and FastAPI

## Architecture

- **Frontend**: Vue.js 3 with modern UI components
- **Backend**: FastAPI with background task processing
- **Processing**: Python script using Google AI for data extraction

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- Google AI API key (configured in the processing script)

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Copy your existing processing files**:
   ```bash
   cp -r ../current ./
   ```

4. **Start the FastAPI server**:
   ```bash
   python main.py
   ```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run serve
   ```

The frontend will be available at `http://localhost:8080`

## Usage

1. **Prepare your files**:
   - Create a ZIP archive containing all patient PDFs (one PDF per patient)
   - Prepare an Excel file with field definitions/instructions

2. **Upload files**:
   - Drag & drop or click to upload the ZIP file
   - Drag & drop or click to upload the Excel file
   - Set the number of pages to extract per patient

3. **Start processing**:
   - Click "Start Processing"
   - Monitor progress in real-time
   - Download the results when complete

## API Endpoints

- `POST /upload` - Upload files and start processing
- `GET /status/{job_id}` - Get processing status
- `GET /download/{job_id}` - Download results
- `GET /health` - Health check

## Deployment

### Backend Deployment (Railway/Render)

1. **Create a new project** on Railway or Render
2. **Connect your repository**
3. **Set environment variables**:
   - `PYTHON_VERSION`: 3.9
4. **Deploy** - the platform will automatically detect FastAPI

### Frontend Deployment (Vercel)

1. **Create a new project** on Vercel
2. **Connect your repository**
3. **Set build settings**:
   - Build Command: `npm run build`
   - Output Directory: `dist`
4. **Set environment variables**:
   - `VUE_APP_API_URL`: Your backend URL
5. **Deploy**

## File Structure

```
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── current/            # Your existing processing scripts
├── frontend/
│   ├── src/
│   │   ├── App.vue         # Main Vue component
│   │   └── main.js         # Vue app entry point
│   ├── public/
│   │   └── index.html      # HTML template
│   ├── package.json        # Node.js dependencies
│   └── vue.config.js       # Vue configuration
└── README.md
```

## Processing Flow

1. **Upload**: ZIP file (patient PDFs) + Excel file (instructions) + page count
2. **Extract**: Unzip PDFs to temporary directory
3. **Process**: Run `2-extract_info.py` for each PDF with specified page count
4. **Combine**: Merge all patient data into single CSV
5. **Download**: Provide results for download

## Configuration

### Page Count
- Default: 2 pages per patient
- Range: 1-50 pages
- Applied to all PDFs in the batch

### Processing Settings
- Max workers: 3 concurrent threads
- Timeout: Configurable per deployment
- Retry logic: Built into the processing script

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure backend CORS settings match your frontend URL
2. **File Upload Failures**: Check file size limits and format validation
3. **Processing Timeouts**: Adjust worker count or timeout settings
4. **API Key Issues**: Verify Google AI API key is configured correctly

### Logs

- Backend logs: Check deployment platform logs
- Frontend errors: Browser developer console
- Processing errors: Backend job status endpoint

## Development

### Adding New Features

1. **Backend**: Extend FastAPI endpoints in `main.py`
2. **Frontend**: Add Vue components and update `App.vue`
3. **Processing**: Modify `2-extract_info.py` for new extraction logic

### Testing

- Backend: Use FastAPI's automatic docs at `/docs`
- Frontend: Run `npm run test` for unit tests
- Integration: Test full workflow with sample files

## License

This project is for internal use. Please ensure compliance with data privacy regulations when processing patient information. 