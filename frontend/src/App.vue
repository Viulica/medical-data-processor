<template>
  <div id="app">
    <header class="header">
      <div class="header-content">
        <div class="logo">
          <div class="logo-icon">🏥</div>
          <div class="logo-text">
            <h1>Medical Data Processor</h1>
            <p>AI-Powered Patient Document Analysis</p>
          </div>
        </div>
      </div>
    </header>

    <main class="main-content">
      <div class="container">
        <!-- Tab Navigation -->
        <div class="tab-navigation">
          <button 
            @click="activeTab = 'process'" 
            :class="{ active: activeTab === 'process' }"
            class="tab-btn"
          >
            📊 Process Documents
          </button>
          <button 
            @click="activeTab = 'split'" 
            :class="{ active: activeTab === 'split' }"
            class="tab-btn"
          >
            ✂️ Split PDF
          </button>
        </div>

        <!-- Process Documents Tab -->
        <div v-if="activeTab === 'process'" class="upload-section">
          <div class="section-header">
            <h2>Document Processing</h2>
            <p>Upload patient documents and processing instructions to extract structured medical data</p>
          </div>

          <div class="upload-grid">
            <!-- Step 1: ZIP File Upload -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">1</div>
                <h3>Patient Documents</h3>
              </div>
              <div 
                class="dropzone"
                :class="{ 
                  'active': isZipDragActive, 
                  'has-file': zipFile 
                }"
                @drop="onZipDrop"
                @dragover.prevent
                @dragenter.prevent
                @click="triggerZipUpload"
              >
                <input 
                  ref="zipInput" 
                  type="file" 
                  accept=".zip" 
                  @change="onZipFileSelect" 
                  style="display: none"
                />
                <div class="upload-content">
                  <div class="upload-icon">📁</div>
                  <div v-if="zipFile" class="file-info">
                    <div class="file-icon">📄</div>
                    <span class="file-name">{{ zipFile.name }}</span>
                    <span class="file-size">{{ formatFileSize(zipFile.size) }}</span>
                  </div>
                  <p v-else class="upload-text">Drag & drop ZIP archive here<br>or click to browse</p>
                </div>
              </div>
            </div>

            <!-- Step 2: Excel File Upload -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">2</div>
                <h3>Processing Template</h3>
              </div>
              <div 
                class="dropzone"
                :class="{ 
                  'active': isExcelDragActive, 
                  'has-file': excelFile 
                }"
                @drop="onExcelDrop"
                @dragover.prevent
                @dragenter.prevent
                @click="triggerExcelUpload"
              >
                <input 
                  ref="excelInput" 
                  type="file" 
                  accept=".xlsx,.xls" 
                  @change="onExcelFileSelect" 
                  style="display: none"
                />
                <div class="upload-content">
                  <div class="upload-icon">📊</div>
                  <div v-if="excelFile" class="file-info">
                    <div class="file-icon">📄</div>
                    <span class="file-name">{{ excelFile.name }}</span>
                    <span class="file-size">{{ formatFileSize(excelFile.size) }}</span>
                  </div>
                  <p v-else class="upload-text">Drag & drop Excel template here<br>or click to browse</p>
                </div>
              </div>
            </div>

            <!-- Step 3: Configuration -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">3</div>
                <h3>Processing Settings</h3>
              </div>
              <div class="settings-content">
                <div class="setting-group">
                  <label for="pageCount">Pages per document</label>
                  <div class="input-wrapper">
                    <input 
                      id="pageCount"
                      v-model.number="pageCount" 
                      type="number" 
                      min="1" 
                      max="50" 
                      class="page-input"
                      placeholder="2"
                    />
                  </div>
                  <small class="help-text">
                    Number of pages to extract from each patient document
                  </small>
                </div>
              </div>
            </div>
          </div>

          <!-- Action Buttons -->
          <div class="action-section">
            <button 
              @click="startProcessing" 
              :disabled="!canProcess || isProcessing"
              class="process-btn"
            >
              <span v-if="isProcessing" class="spinner"></span>
              <span v-else class="btn-icon">🚀</span>
              {{ isProcessing ? 'Processing Documents...' : 'Start Processing' }}
            </button>
            
            <button 
              v-if="zipFile || excelFile || jobId" 
              @click="resetForm" 
              class="reset-btn"
            >
              Reset
            </button>
          </div>
        </div>

        <!-- Processing Status -->
        <div v-if="jobStatus" class="status-section">
          <div class="status-card">
            <div class="status-header">
              <div class="status-indicator" :class="jobStatus.status">
                <span class="status-icon">{{ getStatusIcon() }}</span>
              </div>
              <div class="status-info">
                <h3>{{ getStatusTitle() }}</h3>
                <p class="status-message">{{ jobStatus.message }}</p>
              </div>
            </div>
            
            <div v-if="jobStatus.status === 'processing'" class="progress-section">
              <div class="progress-bar">
                <div 
                  class="progress-fill" 
                  :style="{ width: `${jobStatus.progress}%` }"
                ></div>
              </div>
              <div class="progress-text">{{ jobStatus.progress }}% Complete</div>
            </div>
            
            <div v-if="jobStatus.status === 'completed'" class="success-section">
              <button @click="downloadResults" class="download-btn">
                <span class="btn-icon">📥</span>
                Download Results (CSV)
              </button>
            </div>
            
            <div v-if="jobStatus.status === 'failed' && jobStatus.error" class="error-section">
              <div class="error-message">
                <span class="error-icon">⚠️</span>
                <span>{{ jobStatus.error }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Split PDF Tab -->
        <div v-if="activeTab === 'split'" class="upload-section">
          <div class="section-header">
            <h2>PDF Splitting</h2>
            <p>Upload a single PDF containing multiple patient documents to split into individual files</p>
          </div>

          <div class="upload-grid">
            <!-- PDF File Upload -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">1</div>
                <h3>Combined PDF</h3>
              </div>
              <div 
                class="dropzone"
                :class="{ 
                  'active': isPdfDragActive, 
                  'has-file': pdfFile 
                }"
                @drop="onPdfDrop"
                @dragover.prevent
                @dragenter.prevent
                @click="triggerPdfUpload"
              >
                <input 
                  ref="pdfInput" 
                  type="file" 
                  accept=".pdf" 
                  @change="onPdfFileSelect" 
                  style="display: none"
                />
                <div class="upload-content">
                  <div class="upload-icon">📄</div>
                  <div v-if="pdfFile" class="file-info">
                    <div class="file-icon">📄</div>
                    <span class="file-name">{{ pdfFile.name }}</span>
                    <span class="file-size">{{ formatFileSize(pdfFile.size) }}</span>
                  </div>
                  <p v-else class="upload-text">Drag & drop PDF file here<br>or click to browse</p>
                </div>
              </div>
            </div>

            <!-- Filter String Input -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">2</div>
                <h3>Split Filter</h3>
              </div>
              <div class="filter-input-section">
                <label for="filterString" class="filter-label">
                  Text to search for in PDF pages:
                </label>
                <input 
                  id="filterString"
                  v-model="filterString"
                  type="text" 
                  placeholder="e.g., Patient Address, Patient Information, Anesthesia Billing, etc."
                  class="filter-input"
                  :disabled="isSplitting"
                />
                <p class="filter-help">
                  Enter the text that appears on pages where you want to split the PDF. 
                  The system will create a new section starting from each page containing this text.
                  <strong>Case insensitive search.</strong>
                </p>
              </div>
            </div>
          </div>

          <!-- Action Buttons -->
          <div class="action-section">
            <button 
              @click="startSplitting" 
              :disabled="!canSplit || isSplitting"
              class="process-btn"
            >
              <span v-if="isSplitting" class="spinner"></span>
              <span v-else class="btn-icon">✂️</span>
              {{ isSplitting ? 'Splitting PDF...' : 'Split PDF' }}
            </button>
            
            <button 
              v-if="pdfFile || splitJobId" 
              @click="resetSplitForm" 
              class="reset-btn"
            >
              Reset
            </button>
          </div>
        </div>

        <!-- Split Processing Status -->
        <div v-if="splitJobStatus" class="status-section">
          <div class="status-card">
            <div class="status-header">
              <div class="status-indicator" :class="splitJobStatus.status">
                <span class="status-icon">{{ getSplitStatusIcon() }}</span>
              </div>
              <div class="status-info">
                <h3>{{ getSplitStatusTitle() }}</h3>
                <p class="status-message">{{ splitJobStatus.message }}</p>
              </div>
            </div>
            
            <div v-if="splitJobStatus.status === 'processing'" class="progress-section">
              <div class="progress-bar">
                <div 
                  class="progress-fill" 
                  :style="{ width: `${splitJobStatus.progress}%` }"
                ></div>
              </div>
              <div class="progress-text">{{ splitJobStatus.progress }}% Complete</div>
            </div>
            
            <div v-if="splitJobStatus.status === 'completed'" class="success-section">
              <button @click="downloadSplitResults" class="download-btn">
                <span class="btn-icon">📥</span>
                Download Split PDFs (ZIP)
              </button>
            </div>
            
            <div v-if="splitJobStatus.status === 'failed' && splitJobStatus.error" class="error-section">
              <div class="error-message">
                <span class="error-icon">⚠️</span>
                <span>{{ splitJobStatus.error }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>

    <footer class="footer">
      <div class="footer-content">
        <p>&copy; 2025 Medical Data Processor. Secure, HIPAA-compliant document processing.</p>
      </div>
    </footer>
  </div>
</template>

<script>
import axios from 'axios'
import { useToast } from 'vue-toastification'

// Helper function to properly join URL parts without double slashes
function joinUrl(base, path) {
  const cleanBase = base.replace(/\/+$/, '') // Remove trailing slashes
  const cleanPath = path.replace(/^\/+/, '') // Remove leading slashes
  return `${cleanBase}/${cleanPath}`
}

const API_BASE_URL = (process.env.VUE_APP_API_URL || 'http://localhost:8000').replace(/\/+$/, '') // Remove all trailing slashes to prevent double slashes

export default {
  name: 'App',
  setup() {
    const toast = useToast()
    return { toast }
  },
  data() {
    return {
      activeTab: 'process',
      zipFile: null,
      excelFile: null,
      pageCount: 2,
      jobId: null,
      jobStatus: null,
      isProcessing: false,
      isZipDragActive: false,
      isExcelDragActive: false,
      pollingInterval: null,
      // Split PDF functionality
      pdfFile: null,
      filterString: '',
      splitJobId: null,
      splitJobStatus: null,
      isSplitting: false,
      isPdfDragActive: false,
      splitPollingInterval: null
    }
  },
  computed: {
    canProcess() {
      return this.zipFile && this.excelFile && this.pageCount >= 1 && this.pageCount <= 50
    },
    canSplit() {
      return this.pdfFile && this.filterString.trim()
    }
  },
  methods: {
    formatFileSize(bytes) {
      if (bytes === 0) return '0 Bytes'
      const k = 1024
      const sizes = ['Bytes', 'KB', 'MB', 'GB']
      const i = Math.floor(Math.log(bytes) / Math.log(k))
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
    },

    getStatusTitle() {
      if (!this.jobStatus) return ''
      
      switch (this.jobStatus.status) {
        case 'completed':
          return 'Processing Complete'
        case 'failed':
          return 'Processing Failed'
        case 'processing':
          return 'Processing Documents'
        default:
          return 'Processing Status'
      }
    },

    // File upload handlers
    onZipDrop(e) {
      e.preventDefault()
      this.isZipDragActive = false
      const files = e.dataTransfer.files
      if (files.length > 0 && files[0].name.endsWith('.zip')) {
        this.zipFile = files[0]
        this.toast.success('Patient documents uploaded successfully!')
      } else {
        this.toast.error('Please upload a valid ZIP archive')
      }
    },

    onExcelDrop(e) {
      e.preventDefault()
      this.isExcelDragActive = false
      const files = e.dataTransfer.files
      if (files.length > 0 && (files[0].name.endsWith('.xlsx') || files[0].name.endsWith('.xls'))) {
        this.excelFile = files[0]
        this.toast.success('Processing template uploaded successfully!')
      } else {
        this.toast.error('Please upload a valid Excel template')
      }
    },

    triggerZipUpload() {
      this.$refs.zipInput.click()
    },

    triggerExcelUpload() {
      this.$refs.excelInput.click()
    },

    onZipFileSelect(e) {
      const file = e.target.files[0]
      if (file && file.name.endsWith('.zip')) {
        this.zipFile = file
        this.toast.success('Patient documents uploaded successfully!')
      } else {
        this.toast.error('Please select a valid ZIP archive')
      }
    },

    onExcelFileSelect(e) {
      const file = e.target.files[0]
      if (file && (file.name.endsWith('.xlsx') || file.name.endsWith('.xls'))) {
        this.excelFile = file
        this.toast.success('Processing template uploaded successfully!')
      } else {
        this.toast.error('Please select a valid Excel template')
      }
    },

    // PDF Split functionality
    onPdfDrop(e) {
      e.preventDefault()
      this.isPdfDragActive = false
      const files = e.dataTransfer.files
      if (files.length > 0 && files[0].name.endsWith('.pdf')) {
        this.pdfFile = files[0]
        this.toast.success('PDF file uploaded successfully!')
      } else {
        this.toast.error('Please upload a valid PDF file')
      }
    },

    triggerPdfUpload() {
      this.$refs.pdfInput.click()
    },

    onPdfFileSelect(e) {
      const file = e.target.files[0]
      if (file && file.name.endsWith('.pdf')) {
        this.pdfFile = file
        this.toast.success('PDF file uploaded successfully!')
      } else {
        this.toast.error('Please select a valid PDF file')
      }
    },

    async startSplitting() {
      if (!this.pdfFile) {
        this.toast.error('Please upload a PDF file')
        return
      }

      if (!this.filterString.trim()) {
        this.toast.error('Please enter a filter string to search for')
        return
      }

      this.isSplitting = true
      this.splitJobStatus = null

      const formData = new FormData()
      formData.append('pdf_file', this.pdfFile)
      formData.append('filter_string', this.filterString.trim())

      const splitUrl = joinUrl(API_BASE_URL, 'split-pdf')
      console.log('🔧 Split URL:', splitUrl)

      try {
        const response = await axios.post(splitUrl, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        })

        this.splitJobId = response.data.job_id
        this.toast.success('PDF splitting started!')
        
        // Start polling for status
        this.splitPollingInterval = setInterval(() => {
          this.checkSplitJobStatus(response.data.job_id)
        }, 2000)

      } catch (error) {
        console.error('Split upload error:', error)
        this.toast.error('Failed to start PDF splitting. Please try again.')
        this.isSplitting = false
      }
    },

    async checkSplitJobStatus(id) {
      try {
        const statusUrl = joinUrl(API_BASE_URL, `status/${id}`)
        const response = await axios.get(statusUrl)
        
        this.splitJobStatus = response.data
        
        if (response.data.status === 'completed' || response.data.status === 'failed') {
          clearInterval(this.splitPollingInterval)
          this.isSplitting = false
          
          if (response.data.status === 'completed') {
            this.toast.success('PDF splitting completed!')
          } else {
            this.toast.error('PDF splitting failed!')
          }
        }
      } catch (error) {
        console.error('Status check error:', error)
        clearInterval(this.splitPollingInterval)
        this.isSplitting = false
        this.toast.error('Failed to check split status')
      }
    },

    async downloadSplitResults() {
      if (!this.splitJobId) return
      
      try {
        const downloadUrl = joinUrl(API_BASE_URL, `download/${this.splitJobId}`)
        const response = await axios.get(downloadUrl, {
          responseType: 'blob'
        })
        
        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        link.setAttribute('download', `split_pdfs_${this.splitJobId}.zip`)
        document.body.appendChild(link)
        link.click()
        link.remove()
        window.URL.revokeObjectURL(url)
        
        this.toast.success('Split PDFs downloaded successfully!')
      } catch (error) {
        console.error('Download error:', error)
        this.toast.error('Failed to download split PDFs')
      }
    },

    resetSplitForm() {
      this.pdfFile = null
      this.filterString = ''
      this.splitJobId = null
      this.splitJobStatus = null
      this.isSplitting = false
      this.isPdfDragActive = false
      
      if (this.splitPollingInterval) {
        clearInterval(this.splitPollingInterval)
        this.splitPollingInterval = null
      }
      
      // Reset file input
      if (this.$refs.pdfInput) {
        this.$refs.pdfInput.value = ''
      }
    },

    getSplitStatusIcon() {
      if (!this.splitJobStatus) return ''
      
      switch (this.splitJobStatus.status) {
        case 'completed':
          return '✅'
        case 'failed':
          return '❌'
        case 'processing':
          return '⏳'
        default:
          return '⏳'
      }
    },

    getSplitStatusTitle() {
      if (!this.splitJobStatus) return ''
      
      switch (this.splitJobStatus.status) {
        case 'completed':
          return 'PDF Splitting Complete'
        case 'failed':
          return 'PDF Splitting Failed'
        case 'processing':
          return 'Splitting PDF'
        default:
          return 'PDF Splitting Status'
      }
    },

    // Processing methods
    async startProcessing() {
      if (!this.canProcess) {
        this.toast.error('Please upload both files and set a valid page count')
        return
      }

      this.isProcessing = true
      this.jobStatus = null

      const formData = new FormData()
      formData.append('zip_file', this.zipFile)
      formData.append('excel_file', this.excelFile)
      formData.append('n_pages', this.pageCount)

      // Debug: Log the URL being used
      const uploadUrl = joinUrl(API_BASE_URL, 'upload')
      console.log('🔧 API_BASE_URL:', API_BASE_URL)
      console.log('🔧 Upload URL:', uploadUrl)

      try {
        const response = await axios.post(uploadUrl, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        })

        this.jobId = response.data.job_id
        this.toast.success('Processing started!')
        
        // Start polling for status
        this.pollingInterval = setInterval(() => {
          this.checkJobStatus(response.data.job_id)
        }, 2000)

      } catch (error) {
        console.error('Upload error:', error)
        console.error('🔧 Error URL:', uploadUrl)
        this.toast.error('Failed to start processing. Please try again.')
        this.isProcessing = false
      }
    },

    async checkJobStatus(id) {
      try {
        const statusUrl = joinUrl(API_BASE_URL, `status/${id}`)
        console.log('🔧 Status URL:', statusUrl)
        const response = await axios.get(statusUrl)
        this.jobStatus = response.data
        
        if (response.data.status === 'completed' || response.data.status === 'failed') {
          clearInterval(this.pollingInterval)
          this.pollingInterval = null
          this.isProcessing = false
          
          if (response.data.status === 'completed') {
            this.toast.success('Processing completed! You can now download the results.')
          } else {
            this.toast.error(`Processing failed: ${response.data.error}`)
          }
        }
      } catch (error) {
        console.error('Status check error:', error)
      }
    },

    async downloadResults() {
      if (!this.jobId) return
      
      try {
        const response = await axios.get(joinUrl(API_BASE_URL, `download/${this.jobId}`), {
          responseType: 'blob',
        })
        
        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        link.setAttribute('download', `patient_data_${this.jobId}.csv`)
        document.body.appendChild(link)
        link.click()
        link.remove()
        window.URL.revokeObjectURL(url)
        
        this.toast.success('Download started!')
      } catch (error) {
        console.error('Download error:', error)
        this.toast.error('Failed to download results')
      }
    },

    resetForm() {
      this.zipFile = null
      this.excelFile = null
      this.pageCount = 2
      this.jobId = null
      this.jobStatus = null
      this.isProcessing = false
      if (this.pollingInterval) {
        clearInterval(this.pollingInterval)
        this.pollingInterval = null
      }
    },

    getStatusIcon() {
      if (!this.jobStatus) return ''
      
      switch (this.jobStatus.status) {
        case 'completed':
          return '✅'
        case 'failed':
          return '❌'
        case 'processing':
          return '⏳'
        default:
          return '⏸️'
      }
    }
  },

  beforeUnmount() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval)
    }
  }
}
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f8fafc;
  color: #1e293b;
  line-height: 1.6;
}

#app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Header */
.header {
  background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
  color: white;
  padding: 2rem 0;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
}

.logo {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.logo-icon {
  font-size: 2.5rem;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  padding: 0.5rem;
  backdrop-filter: blur(10px);
}

.logo-text h1 {
  font-size: 2rem;
  font-weight: 700;
  margin-bottom: 0.25rem;
}

.logo-text p {
  font-size: 1rem;
  opacity: 0.9;
  font-weight: 400;
}

/* Main Content */
.main-content {
  flex: 1;
  padding: 3rem 0;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
}

/* Tab Navigation */
.tab-navigation {
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-bottom: 3rem;
}

.tab-btn {
  background: white;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  padding: 1rem 2rem;
  font-size: 1rem;
  font-weight: 600;
  color: #64748b;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.tab-btn:hover {
  border-color: #3b82f6;
  color: #3b82f6;
  transform: translateY(-1px);
}

.tab-btn.active {
  background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
  border-color: #3b82f6;
  color: white;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

/* Section Headers */
.section-header {
  text-align: center;
  margin-bottom: 3rem;
}

.section-header h2 {
  font-size: 2.5rem;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 1rem;
}

.section-header p {
  font-size: 1.125rem;
  color: #64748b;
  max-width: 600px;
  margin: 0 auto;
}

/* Upload Grid */
.upload-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 2rem;
  margin-bottom: 3rem;
}

/* Upload Cards */
.upload-card {
  background: white;
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  border: 1px solid #e2e8f0;
  transition: all 0.3s ease;
}

.upload-card:hover {
  box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}

.card-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.step-number {
  background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
  color: white;
  width: 2rem;
  height: 2rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 0.875rem;
}

.card-header h3 {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1e293b;
}

/* Dropzone */
.dropzone {
  border: 2px dashed #cbd5e1;
  border-radius: 12px;
  padding: 2rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: #f8fafc;
  min-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.dropzone:hover {
  border-color: #3b82f6;
  background: #eff6ff;
}

.dropzone.active {
  border-color: #3b82f6;
  background: #dbeafe;
}

.dropzone.has-file {
  border-color: #10b981;
  background: #ecfdf5;
}

.upload-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
}

.upload-icon {
  font-size: 3rem;
  opacity: 0.7;
}

.upload-text {
  color: #64748b;
  font-size: 1rem;
  line-height: 1.5;
}

.file-info {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.file-icon {
  font-size: 2rem;
  color: #10b981;
}

.file-name {
  font-weight: 500;
  color: #1e293b;
  font-size: 0.875rem;
}

.file-size {
  color: #64748b;
  font-size: 0.75rem;
}

/* Settings */
.settings-content {
  padding: 1rem 0;
}

.setting-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.setting-group label {
  font-weight: 500;
  color: #374151;
  font-size: 0.875rem;
}

.input-wrapper {
  position: relative;
  display: flex;
  align-items: center;
}

.page-input {
  flex: 1;
  padding: 0.75rem 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 500;
  transition: all 0.3s ease;
  background: white;
}

.page-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.help-text {
  color: #64748b;
  font-size: 0.75rem;
  line-height: 1.4;
}

/* Action Section */
.action-section {
  display: flex;
  gap: 1rem;
  justify-content: center;
  align-items: center;
  flex-wrap: wrap;
}

.process-btn, .reset-btn {
  padding: 1rem 2rem;
  border: none;
  border-radius: 12px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  min-width: 200px;
  justify-content: center;
}

.process-btn {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  color: white;
  box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);
}

.process-btn:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 10px 25px -3px rgba(16, 185, 129, 0.4);
}

.process-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}

.reset-btn {
  background: white;
  color: #64748b;
  border: 2px solid #e2e8f0;
}

.reset-btn:hover {
  background: #f1f5f9;
  border-color: #cbd5e1;
}

.btn-icon {
  font-size: 1.25rem;
}

.spinner {
  width: 1.25rem;
  height: 1.25rem;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top: 2px solid white;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Status Section */
.status-section {
  margin-top: 3rem;
}

.status-card {
  background: white;
  border-radius: 16px;
  padding: 2rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  border: 1px solid #e2e8f0;
}

.status-header {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.status-indicator {
  width: 4rem;
  height: 4rem;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
}

.status-indicator.completed {
  background: #ecfdf5;
  color: #10b981;
}

.status-indicator.failed {
  background: #fef2f2;
  color: #ef4444;
}

.status-indicator.processing {
  background: #fffbeb;
  color: #f59e0b;
}

.status-info h3 {
  font-size: 1.5rem;
  font-weight: 600;
  color: #1e293b;
  margin-bottom: 0.5rem;
}

.status-message {
  color: #64748b;
  font-size: 1rem;
}

/* Progress Section */
.progress-section {
  margin-bottom: 2rem;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background: #e2e8f0;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 1rem;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3b82f6 0%, #1e40af 100%);
  transition: width 0.3s ease;
  border-radius: 4px;
}

.progress-text {
  text-align: center;
  font-weight: 500;
  color: #374151;
}

/* Success Section */
.success-section {
  text-align: center;
}

.download-btn {
  background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
  color: white;
  border: none;
  padding: 1rem 2rem;
  border-radius: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  display: inline-flex;
  align-items: center;
  gap: 0.75rem;
  box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
}

.download-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 25px -3px rgba(59, 130, 246, 0.4);
}

/* Error Section */
.error-section {
  text-align: center;
}

.error-message {
  display: inline-flex;
  align-items: center;
  gap: 0.75rem;
  color: #ef4444;
  background: #fef2f2;
  padding: 1rem 1.5rem;
  border-radius: 8px;
  border: 1px solid #fecaca;
  font-weight: 500;
}

.error-icon {
  font-size: 1.25rem;
}

/* Footer */
.footer {
  background: #1e293b;
  color: #94a3b8;
  padding: 2rem 0;
  margin-top: auto;
}

.footer-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 2rem;
  text-align: center;
}

.footer-content p {
  font-size: 0.875rem;
}

/* Filter Input Styles */
.filter-input-section {
  padding: 1rem;
}

.filter-label {
  display: block;
  font-weight: 600;
  color: #374151;
  margin-bottom: 0.75rem;
  font-size: 0.875rem;
}

.filter-input {
  width: 100%;
  padding: 0.75rem 1rem;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  font-size: 1rem;
  transition: all 0.3s ease;
  background: white;
}

.filter-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.filter-input:disabled {
  background: #f9fafb;
  color: #9ca3af;
  cursor: not-allowed;
}

.filter-help {
  margin-top: 0.75rem;
  font-size: 0.875rem;
  color: #6b7280;
  line-height: 1.5;
}

/* Responsive Design */
@media (max-width: 768px) {
  .header-content {
    padding: 0 1rem;
  }
  
  .logo {
    flex-direction: column;
    text-align: center;
    gap: 1rem;
  }
  
  .logo-text h1 {
    font-size: 1.5rem;
  }
  
  .container {
    padding: 0 1rem;
  }
  
  .section-header h2 {
    font-size: 2rem;
  }
  
  .upload-grid {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }
  
  .upload-card {
    padding: 1.5rem;
  }
  
  .action-section {
    flex-direction: column;
  }
  
  .process-btn, .reset-btn {
    width: 100%;
  }
  
  .status-header {
    flex-direction: column;
    text-align: center;
    gap: 1rem;
  }
}
</style> 