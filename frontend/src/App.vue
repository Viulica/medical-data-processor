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
            class="tab-btn disabled"
            disabled
          >
            📊 Process Documents (Disabled)
          </button>
          <button
            @click="activeTab = 'process-fast'"
            :class="{ active: activeTab === 'process-fast' }"
            class="tab-btn"
          >
            ⚡ Process Documents (Fast)
          </button>
          <button
            @click="activeTab = 'split'"
            :class="{ active: activeTab === 'split' }"
            class="tab-btn"
          >
            ✂️ Split PDF
          </button>
          <button
            @click="activeTab = 'cpt'"
            :class="{ active: activeTab === 'cpt' }"
            class="tab-btn"
          >
            🏥 Predict CPT Codes
          </button>
          <button
            @click="activeTab = 'uni'"
            :class="{ active: activeTab === 'uni' }"
            class="tab-btn"
          >
            🔄 Convert UNI CSV
          </button>
        </div>

        <!-- Process Documents Tab -->
        <div v-if="activeTab === 'process'" class="upload-section">
          <div class="section-header">
            <h2>Document Processing (Disabled)</h2>
            <p>
              This processing option has been temporarily disabled. Please use
              the "Process Documents (Fast)" option instead.
            </p>
          </div>

          <div class="disabled-section">
            <div class="disabled-card">
              <div class="disabled-icon">🚫</div>
              <h3>Standard Processing Disabled</h3>
              <p>
                The standard document processing feature has been temporarily
                disabled. Please use the "Process Documents (Fast)" tab for
                document processing.
              </p>
              <div class="disabled-features">
                <div class="feature-item">
                  <span class="feature-icon">⚡</span>
                  <span>Use Fast Processing instead</span>
                </div>
                <div class="feature-item">
                  <span class="feature-icon">🔧</span>
                  <span>Same functionality, better performance</span>
                </div>
                <div class="feature-item">
                  <span class="feature-icon">💰</span>
                  <span>More cost-effective processing</span>
                </div>
              </div>
            </div>
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

            <div
              v-if="jobStatus.status === 'processing'"
              class="progress-section"
            >
              <div class="progress-bar">
                <div
                  class="progress-fill"
                  :style="{ width: `${jobStatus.progress}%` }"
                ></div>
              </div>
              <div class="progress-text">
                {{ jobStatus.progress }}% Complete
              </div>
              <button @click="checkJobStatus" class="check-status-btn">
                <span class="btn-icon">🔄</span>
                Check Status
              </button>
            </div>

            <div
              v-if="jobStatus.status === 'completed'"
              class="success-section"
            >
              <button @click="downloadResults" class="download-btn">
                <span class="btn-icon">📥</span>
                Download Results (CSV)
              </button>
            </div>

            <div
              v-if="jobStatus.status === 'failed' && jobStatus.error"
              class="error-section"
            >
              <div class="error-message">
                <span class="error-icon">⚠️</span>
                <span>{{ jobStatus.error }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Process Documents (Fast) Tab -->
        <div v-if="activeTab === 'process-fast'" class="upload-section">
          <div class="section-header">
            <h2>Document Processing (Fast)</h2>
            <p>
              Upload patient documents and processing instructions to extract
              structured medical data using Gemini 2.5 Flash (faster and more
              cost-effective)
            </p>
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
                  active: isZipDragActiveFast,
                  'has-file': zipFileFast,
                }"
                @drop="onZipDropFast"
                @dragover.prevent
                @dragenter.prevent
                @click="triggerZipUploadFast"
              >
                <input
                  ref="zipInputFast"
                  type="file"
                  accept=".zip"
                  @change="onZipFileSelectFast"
                  style="display: none"
                />
                <div class="upload-content">
                  <div class="upload-icon">📁</div>
                  <div v-if="zipFileFast" class="file-info">
                    <div class="file-icon">📄</div>
                    <span class="file-name">{{ zipFileFast.name }}</span>
                    <span class="file-size">{{
                      formatFileSize(zipFileFast.size)
                    }}</span>
                  </div>
                  <p v-else class="upload-text">
                    Drag & drop ZIP archive here<br />or click to browse
                  </p>
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
                  active: isExcelDragActiveFast,
                  'has-file': excelFileFast,
                }"
                @drop="onExcelDropFast"
                @dragover.prevent
                @dragenter.prevent
                @click="triggerExcelUploadFast"
              >
                <input
                  ref="excelInputFast"
                  type="file"
                  accept=".xlsx,.xls"
                  @change="onExcelFileSelectFast"
                  style="display: none"
                />
                <div class="upload-content">
                  <div class="upload-icon">📊</div>
                  <div v-if="excelFileFast" class="file-info">
                    <div class="file-icon">📄</div>
                    <span class="file-name">{{ excelFileFast.name }}</span>
                    <span class="file-size">{{
                      formatFileSize(excelFileFast.size)
                    }}</span>
                  </div>
                  <p v-else class="upload-text">
                    Drag & drop Excel template here<br />or click to browse
                  </p>
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
                  <label for="pageCountFast">Pages per document</label>
                  <div class="input-wrapper">
                    <input
                      id="pageCountFast"
                      v-model.number="pageCountFast"
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
              @click="startProcessingFast"
              :disabled="!canProcessFast || isProcessingFast"
              class="process-btn"
            >
              <span v-if="isProcessingFast" class="spinner"></span>
              <span v-else class="btn-icon">⚡</span>
              {{
                isProcessingFast
                  ? "Processing Documents..."
                  : "Start Fast Processing"
              }}
            </button>

            <button
              v-if="zipFileFast || excelFileFast || jobIdFast"
              @click="resetFormFast"
              class="reset-btn"
            >
              Reset
            </button>
          </div>
        </div>

        <!-- Processing Status (Fast) -->
        <div v-if="jobStatusFast" class="status-section">
          <div class="status-card">
            <div class="status-header">
              <div class="status-indicator" :class="jobStatusFast.status">
                <span class="status-icon">{{ getStatusIconFast() }}</span>
              </div>
              <div class="status-info">
                <h3>{{ getStatusTitleFast() }}</h3>
                <p class="status-message">{{ jobStatusFast.message }}</p>
              </div>
            </div>

            <div
              v-if="jobStatusFast.status === 'processing'"
              class="progress-section"
            >
              <div class="progress-bar">
                <div
                  class="progress-fill"
                  :style="{ width: `${jobStatusFast.progress}%` }"
                ></div>
              </div>
              <div class="progress-text">
                {{ jobStatusFast.progress }}% Complete
              </div>
              <button @click="checkJobStatusFast" class="check-status-btn">
                <span class="btn-icon">🔄</span>
                Check Status
              </button>
            </div>

            <div
              v-if="jobStatusFast.status === 'completed'"
              class="success-section"
            >
              <button @click="downloadResultsFast" class="download-btn">
                <span class="btn-icon">📥</span>
                Download Results (CSV)
              </button>
            </div>

            <div
              v-if="jobStatusFast.status === 'failed' && jobStatusFast.error"
              class="error-section"
            >
              <div class="error-message">
                <span class="error-icon">⚠️</span>
                <span>{{ jobStatusFast.error }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Split PDF Tab -->
        <div v-if="activeTab === 'split'" class="upload-section">
          <div class="section-header">
            <h2>PDF Splitting</h2>
            <p>
              Upload a single PDF containing multiple patient documents to split
              into individual files
            </p>
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
                  active: isPdfDragActive,
                  'has-file': pdfFile,
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
                    <span class="file-size">{{
                      formatFileSize(pdfFile.size)
                    }}</span>
                  </div>
                  <p v-else class="upload-text">
                    Drag & drop PDF file here<br />or click to browse
                  </p>
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
                  Enter the text that appears on pages where you want to split
                  the PDF. The system will create a new section starting from
                  each page containing this text.
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
              {{ isSplitting ? "Splitting PDF..." : "Split PDF" }}
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

            <div
              v-if="splitJobStatus.status === 'processing'"
              class="progress-section"
            >
              <div class="progress-bar">
                <div
                  class="progress-fill"
                  :style="{ width: `${splitJobStatus.progress}%` }"
                ></div>
              </div>
              <div class="progress-text">
                {{ splitJobStatus.progress }}% Complete
              </div>
              <p class="processing-note">
                Processing in progress... This may take several minutes.
              </p>
            </div>

            <div
              v-if="splitJobStatus.status === 'completed'"
              class="success-section"
            >
              <button @click="downloadSplitResults" class="download-btn">
                <span class="btn-icon">📥</span>
                Download Split PDFs (ZIP)
              </button>
            </div>

            <div
              v-if="splitJobStatus.status === 'failed' && splitJobStatus.error"
              class="error-section"
            >
              <div class="error-message">
                <span class="error-icon">⚠️</span>
                <span>{{ splitJobStatus.error }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- CPT Prediction Tab -->
        <div v-if="activeTab === 'cpt'" class="upload-section">
          <div class="section-header">
            <h2>CPT Code Prediction</h2>
            <p>
              Upload a CSV file with a 'Procedure' column to predict anesthesia
              CPT codes using AI
            </p>
          </div>

          <div class="upload-grid">
            <!-- CSV File Upload -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">1</div>
                <h3>CSV File with Procedures</h3>
              </div>
              <div
                class="dropzone"
                :class="{
                  active: isCsvDragActive,
                  'has-file': csvFile,
                }"
                @drop="onCsvDrop"
                @dragover.prevent
                @dragenter.prevent
                @click="triggerCsvUpload"
              >
                <input
                  ref="csvInput"
                  type="file"
                  accept=".csv"
                  @change="onCsvFileSelect"
                  style="display: none"
                />
                <div class="upload-content">
                  <div class="upload-icon">📊</div>
                  <div v-if="csvFile" class="file-info">
                    <div class="file-icon">📄</div>
                    <span class="file-name">{{ csvFile.name }}</span>
                    <span class="file-size">{{
                      formatFileSize(csvFile.size)
                    }}</span>
                  </div>
                  <p v-else class="upload-text">
                    Drag & drop CSV file here<br />or click to browse
                  </p>
                </div>
              </div>
            </div>

            <!-- Client Selection -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">2</div>
                <h3>Select Client</h3>
              </div>
              <div class="settings-content">
                <div class="form-group">
                  <label for="client-select" class="form-label"
                    >Choose Client for CPT Coding:</label
                  >
                  <select
                    id="client-select"
                    v-model="selectedClient"
                    class="form-select"
                  >
                    <option value="uni">UNI</option>
                    <option value="sio-stl">SIO-STL</option>
                    <option value="gap-fin">GAP-FIN</option>
                    <option value="apo-utp">APO-UTP</option>
                  </select>
                </div>
              </div>
            </div>

            <!-- Instructions -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">3</div>
                <h3>Requirements</h3>
              </div>
              <div class="settings-content">
                <div class="requirement-list">
                  <div class="requirement-item">
                    <span class="requirement-icon">📋</span>
                    <span>CSV must contain a 'Procedure' column</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">🏥</span>
                    <span>Each row should have a procedure description</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">⚡</span>
                    <span>AI will predict anesthesia CPT codes</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Action Buttons -->
          <div class="action-section">
            <button
              @click="startCptPrediction"
              :disabled="!canPredictCpt || isPredictingCpt"
              class="process-btn"
            >
              <span v-if="isPredictingCpt" class="spinner"></span>
              <span v-else class="btn-icon">🏥</span>
              {{
                isPredictingCpt
                  ? "Predicting CPT Codes..."
                  : "Start CPT Prediction"
              }}
            </button>

            <button
              v-if="csvFile || cptJobId"
              @click="resetCptForm"
              class="reset-btn"
            >
              Reset
            </button>
          </div>
        </div>

        <!-- CPT Prediction Status -->
        <div v-if="cptJobStatus" class="status-section">
          <div class="status-card">
            <div class="status-header">
              <div class="status-indicator" :class="cptJobStatus.status">
                <span class="status-icon">{{ getCptStatusIcon() }}</span>
              </div>
              <div class="status-info">
                <h3>{{ getCptStatusTitle() }}</h3>
                <p class="status-message">{{ cptJobStatus.message }}</p>
              </div>
            </div>

            <div
              v-if="cptJobStatus.status === 'processing'"
              class="progress-section"
            >
              <div class="progress-bar">
                <div
                  class="progress-fill"
                  :style="{ width: `${cptJobStatus.progress}%` }"
                ></div>
              </div>
              <div class="progress-text">
                {{ cptJobStatus.progress }}% Complete
              </div>
              <button @click="checkCptJobStatus" class="check-status-btn">
                <span class="btn-icon">🔄</span>
                Check Status
              </button>
            </div>

            <div
              v-if="cptJobStatus.status === 'completed'"
              class="success-section"
            >
              <button @click="downloadCptResults" class="download-btn">
                <span class="btn-icon">📥</span>
                Download Results (CSV)
              </button>
            </div>

            <div
              v-if="cptJobStatus.status === 'failed' && cptJobStatus.error"
              class="error-section"
            >
              <div class="error-message">
                <span class="error-icon">⚠️</span>
                <span>{{ cptJobStatus.error }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- UNI Conversion Tab -->
        <div v-if="activeTab === 'uni'" class="upload-section">
          <div class="section-header">
            <h2>UNI CSV Conversion</h2>
            <p>
              Upload a UNI CSV file to convert it using the automated conversion
              script
            </p>
          </div>

          <div class="upload-grid">
            <!-- CSV File Upload -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">1</div>
                <h3>UNI CSV File</h3>
              </div>
              <div
                class="dropzone"
                :class="{
                  active: isUniCsvDragActive,
                  'has-file': uniCsvFile,
                }"
                @drop="onUniCsvDrop"
                @dragover.prevent
                @dragenter.prevent
                @click="triggerUniCsvUpload"
              >
                <input
                  ref="uniCsvInput"
                  type="file"
                  accept=".csv"
                  @change="onUniCsvFileSelect"
                  style="display: none"
                />
                <div class="upload-content">
                  <div class="upload-icon">📊</div>
                  <div v-if="uniCsvFile" class="file-info">
                    <div class="file-icon">📄</div>
                    <span class="file-name">{{ uniCsvFile.name }}</span>
                    <span class="file-size">{{
                      formatFileSize(uniCsvFile.size)
                    }}</span>
                  </div>
                  <p v-else class="upload-text">
                    Drag & drop UNI CSV file here<br />or click to browse
                  </p>
                </div>
              </div>
            </div>

            <!-- Instructions -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">2</div>
                <h3>Requirements</h3>
              </div>
              <div class="settings-content">
                <div class="requirement-list">
                  <div class="requirement-item">
                    <span class="requirement-icon">📋</span>
                    <span>CSV file with UNI data format</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">🔄</span>
                    <span>Automatic conversion using mapping rules</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">📥</span>
                    <span>Download converted CSV file</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Action Buttons -->
          <div class="action-section">
            <button
              @click="startUniConversion"
              :disabled="!canConvertUni || isConvertingUni"
              class="process-btn"
            >
              <span v-if="isConvertingUni" class="spinner"></span>
              <span v-else class="btn-icon">🔄</span>
              {{
                isConvertingUni
                  ? "Converting UNI CSV..."
                  : "Start UNI Conversion"
              }}
            </button>

            <button
              v-if="uniCsvFile || uniJobId"
              @click="resetUniForm"
              class="reset-btn"
            >
              Reset
            </button>
          </div>
        </div>

        <!-- UNI Conversion Status -->
        <div v-if="uniJobStatus" class="status-section">
          <div class="status-card">
            <div class="status-header">
              <div class="status-indicator" :class="uniJobStatus.status">
                <span class="status-icon">{{ getUniStatusIcon() }}</span>
              </div>
              <div class="status-info">
                <h3>{{ getUniStatusTitle() }}</h3>
                <p class="status-message">{{ uniJobStatus.message }}</p>
              </div>
            </div>

            <div
              v-if="uniJobStatus.status === 'processing'"
              class="progress-section"
            >
              <div class="progress-bar">
                <div
                  class="progress-fill"
                  :style="{ width: `${uniJobStatus.progress}%` }"
                ></div>
              </div>
              <div class="progress-text">
                {{ uniJobStatus.progress }}% Complete
              </div>
              <button @click="checkUniJobStatus" class="check-status-btn">
                <span class="btn-icon">🔄</span>
                Check Status
              </button>
            </div>

            <div
              v-if="uniJobStatus.status === 'completed'"
              class="success-section"
            >
              <button @click="downloadUniResults" class="download-btn">
                <span class="btn-icon">📥</span>
                Download Converted CSV
              </button>
            </div>

            <div
              v-if="uniJobStatus.status === 'failed' && uniJobStatus.error"
              class="error-section"
            >
              <div class="error-message">
                <span class="error-icon">⚠️</span>
                <span>{{ uniJobStatus.error }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>

    <footer class="footer">
      <div class="footer-content">
        <p>
          &copy; 2025 Medical Data Processor. Secure, HIPAA-compliant document
          processing.
        </p>
      </div>
    </footer>
  </div>
</template>

<script>
import axios from "axios";
import { useToast } from "vue-toastification";

// Helper function to properly join URL parts without double slashes
function joinUrl(base, path) {
  const cleanBase = base.replace(/\/+$/, ""); // Remove trailing slashes
  const cleanPath = path.replace(/^\/+/, ""); // Remove leading slashes
  return `${cleanBase}/${cleanPath}`;
}

const API_BASE_URL = (
  process.env.VUE_APP_API_URL || "http://localhost:8000"
).replace(/\/+$/, ""); // Remove all trailing slashes to prevent double slashes

export default {
  name: "App",
  setup() {
    const toast = useToast();
    return { toast };
  },
  data() {
    return {
      activeTab: "process-fast",
      zipFile: null,
      excelFile: null,
      pageCount: 2,
      jobId: null,
      jobStatus: null,
      isProcessing: false,
      isZipDragActive: false,
      isExcelDragActive: false,
      // Fast processing functionality
      zipFileFast: null,
      excelFileFast: null,
      pageCountFast: 2,
      jobIdFast: null,
      jobStatusFast: null,
      isProcessingFast: false,
      isZipDragActiveFast: false,
      isExcelDragActiveFast: false,
      // Split PDF functionality
      pdfFile: null,
      filterString: "",
      splitJobId: null,
      splitJobStatus: null,
      isSplitting: false,
      isPdfDragActive: false,
      statusPollingInterval: null,
      // CPT prediction functionality
      csvFile: null,
      selectedClient: "uni",
      cptJobId: null,
      cptJobStatus: null,
      isPredictingCpt: false,
      isCsvDragActive: false,
      // UNI conversion functionality
      uniCsvFile: null,
      uniJobId: null,
      uniJobStatus: null,
      isConvertingUni: false,
      isUniCsvDragActive: false,
    };
  },
  computed: {
    canProcess() {
      return (
        this.zipFile &&
        this.excelFile &&
        this.pageCount >= 1 &&
        this.pageCount <= 50
      );
    },
    canProcessFast() {
      return (
        this.zipFileFast &&
        this.excelFileFast &&
        this.pageCountFast >= 1 &&
        this.pageCountFast <= 50
      );
    },
    canSplit() {
      return this.pdfFile && this.filterString.trim();
    },
    canPredictCpt() {
      return this.csvFile;
    },
    canConvertUni() {
      return this.uniCsvFile;
    },
  },
  methods: {
    formatFileSize(bytes) {
      if (bytes === 0) return "0 Bytes";
      const k = 1024;
      const sizes = ["Bytes", "KB", "MB", "GB"];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    },

    getStatusTitle() {
      if (!this.jobStatus) return "";

      switch (this.jobStatus.status) {
        case "completed":
          return "Processing Complete";
        case "failed":
          return "Processing Failed";
        case "processing":
          return "Processing Documents";
        default:
          return "Processing Status";
      }
    },

    // File upload handlers
    onZipDrop(e) {
      e.preventDefault();
      this.isZipDragActive = false;
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].name.endsWith(".zip")) {
        this.zipFile = files[0];
        this.toast.success("Patient documents uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid ZIP archive");
      }
    },

    onExcelDrop(e) {
      e.preventDefault();
      this.isExcelDragActive = false;
      const files = e.dataTransfer.files;
      if (
        files.length > 0 &&
        (files[0].name.endsWith(".xlsx") || files[0].name.endsWith(".xls"))
      ) {
        this.excelFile = files[0];
        this.toast.success("Processing template uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid Excel template");
      }
    },

    triggerZipUpload() {
      this.$refs.zipInput.click();
    },

    triggerExcelUpload() {
      this.$refs.excelInput.click();
    },

    onZipFileSelect(e) {
      const file = e.target.files[0];
      if (file && file.name.endsWith(".zip")) {
        this.zipFile = file;
        this.toast.success("Patient documents uploaded successfully!");
      } else {
        this.toast.error("Please select a valid ZIP archive");
      }
    },

    onExcelFileSelect(e) {
      const file = e.target.files[0];
      if (file && (file.name.endsWith(".xlsx") || file.name.endsWith(".xls"))) {
        this.excelFile = file;
        this.toast.success("Processing template uploaded successfully!");
      } else {
        this.toast.error("Please select a valid Excel template");
      }
    },

    // PDF Split functionality
    onPdfDrop(e) {
      e.preventDefault();
      this.isPdfDragActive = false;
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].name.endsWith(".pdf")) {
        this.pdfFile = files[0];
        this.toast.success("PDF file uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid PDF file");
      }
    },

    triggerPdfUpload() {
      this.$refs.pdfInput.click();
    },

    onPdfFileSelect(e) {
      const file = e.target.files[0];
      if (file && file.name.endsWith(".pdf")) {
        this.pdfFile = file;
        this.toast.success("PDF file uploaded successfully!");
      } else {
        this.toast.error("Please select a valid PDF file");
      }
    },

    async startSplitting() {
      console.log("🚀 Starting PDF splitting process...");
      console.log(
        "📄 PDF File:",
        this.pdfFile
          ? {
              name: this.pdfFile.name,
              size: this.pdfFile.size,
              type: this.pdfFile.type,
            }
          : "No file"
      );
      console.log("🔍 Filter String:", this.filterString);

      if (!this.pdfFile) {
        console.error("❌ No PDF file selected");
        this.toast.error("Please upload a PDF file");
        return;
      }

      if (!this.filterString.trim()) {
        console.error("❌ No filter string provided");
        this.toast.error("Please enter a filter string to search for");
        return;
      }

      this.isSplitting = true;

      const formData = new FormData();
      formData.append("pdf_file", this.pdfFile);
      formData.append("filter_string", this.filterString.trim());

      const splitUrl = joinUrl(API_BASE_URL, "split-pdf");
      console.log("🔧 Split URL:", splitUrl);

      try {
        console.log("📤 Sending split request to backend...");
        const response = await axios.post(splitUrl, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          timeout: 300000, // 5 minute timeout
        });

        console.log("✅ Split request successful:", response.data);
        this.toast.success(
          "PDF splitting started! Check the download section below."
        );

        // Store the job ID for status checking
        this.splitJobId = response.data.job_id;
        this.splitJobStatus = {
          status: "processing",
          progress: 0,
          message: "Processing started...",
        };

        // Start automatic status checking
        this.startStatusPolling();
      } catch (error) {
        console.error("❌ Split processing error:", error);
        console.error("❌ Error details:", {
          message: error.message,
          status: error.response?.status,
          statusText: error.response?.statusText,
          data: error.response?.data,
        });

        let errorMessage = "Failed to start PDF splitting. Please try again.";
        if (error.response?.data?.detail) {
          errorMessage = `Server error: ${error.response.data.detail}`;
        } else if (error.code === "NETWORK_ERROR") {
          errorMessage = "Network error: Unable to connect to the server";
        } else if (error.code === "ECONNABORTED") {
          errorMessage = "Request timeout: Server took too long to respond";
        }

        this.toast.error(errorMessage);
        this.isSplitting = false;
      }
    },

    startStatusPolling() {
      // Check status every 10 seconds
      this.statusPollingInterval = setInterval(async () => {
        if (!this.splitJobId) {
          clearInterval(this.statusPollingInterval);
          return;
        }

        try {
          const statusUrl = joinUrl(API_BASE_URL, `status/${this.splitJobId}`);
          const response = await axios.get(statusUrl);

          this.splitJobStatus = response.data;

          if (response.data.status === "completed") {
            clearInterval(this.statusPollingInterval);
            this.toast.success("PDF splitting completed!");
            this.isSplitting = false;
          } else if (response.data.status === "failed") {
            clearInterval(this.statusPollingInterval);
            this.toast.error(
              `PDF splitting failed: ${response.data.error || "Unknown error"}`
            );
            this.isSplitting = false;
          }
        } catch (error) {
          console.error("Status check error:", error);
        }
      }, 10000); // Check every 10 seconds
    },

    async checkSplitStatus() {
      if (!this.splitJobId) {
        this.toast.error("No job ID available");
        return;
      }

      try {
        const statusUrl = joinUrl(API_BASE_URL, `status/${this.splitJobId}`);
        const response = await axios.get(statusUrl);

        this.splitJobStatus = response.data;

        if (response.data.status === "completed") {
          this.toast.success("PDF splitting completed!");
          this.isSplitting = false;
        } else if (response.data.status === "failed") {
          this.toast.error(
            `PDF splitting failed: ${response.data.error || "Unknown error"}`
          );
          this.isSplitting = false;
        }
      } catch (error) {
        console.error("Status check error:", error);
        this.toast.error("Failed to check status");
      }
    },

    async downloadSplitResults() {
      if (!this.splitJobId) return;

      try {
        const downloadUrl = joinUrl(
          API_BASE_URL,
          `download/${this.splitJobId}`
        );
        const response = await axios.get(downloadUrl, {
          responseType: "blob",
        });

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute("download", `split_pdfs_${this.splitJobId}.zip`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        this.toast.success("Split PDFs downloaded successfully!");
      } catch (error) {
        console.error("Download error:", error);
        this.toast.error("Failed to download split PDFs");
      }
    },

    resetSplitForm() {
      this.pdfFile = null;
      this.filterString = "";
      this.splitJobId = null;
      this.splitJobStatus = null;
      this.isSplitting = false;
      this.isPdfDragActive = false;

      // Clear polling interval
      if (this.statusPollingInterval) {
        clearInterval(this.statusPollingInterval);
        this.statusPollingInterval = null;
      }

      // Reset file input
      if (this.$refs.pdfInput) {
        this.$refs.pdfInput.value = "";
      }
    },

    getSplitStatusIcon() {
      if (!this.splitJobStatus) return "";

      switch (this.splitJobStatus.status) {
        case "completed":
          return "✅";
        case "failed":
          return "❌";
        case "processing":
          return "⏳";
        default:
          return "⏳";
      }
    },

    getSplitStatusTitle() {
      if (!this.splitJobStatus) return "";

      switch (this.splitJobStatus.status) {
        case "completed":
          return "PDF Splitting Complete";
        case "failed":
          return "PDF Splitting Failed";
        case "processing":
          return "Splitting PDF";
        default:
          return "PDF Splitting Status";
      }
    },

    // Processing methods
    async startProcessing() {
      if (!this.canProcess) {
        this.toast.error("Please upload both files and set a valid page count");
        return;
      }

      this.isProcessing = true;
      this.jobStatus = null;

      const formData = new FormData();
      formData.append("zip_file", this.zipFile);
      formData.append("excel_file", this.excelFile);
      formData.append("n_pages", this.pageCount);

      // Debug: Log the URL being used
      const uploadUrl = joinUrl(API_BASE_URL, "upload");
      console.log("🔧 API_BASE_URL:", API_BASE_URL);
      console.log("🔧 Upload URL:", uploadUrl);

      try {
        const response = await axios.post(uploadUrl, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        this.jobId = response.data.job_id;
        this.toast.success(
          "Processing started! Check the status section below."
        );

        // Set initial status
        this.jobStatus = {
          status: "processing",
          progress: 0,
          message: "Processing started...",
        };
      } catch (error) {
        console.error("Upload error:", error);
        console.error("🔧 Error URL:", uploadUrl);
        this.toast.error("Failed to start processing. Please try again.");
        this.isProcessing = false;
      }
    },

    async checkJobStatus() {
      if (!this.jobId) {
        this.toast.error("No job ID available");
        return;
      }

      try {
        const statusUrl = joinUrl(API_BASE_URL, `status/${this.jobId}`);
        const response = await axios.get(statusUrl);

        this.jobStatus = response.data;

        if (response.data.status === "completed") {
          this.toast.success("Processing completed!");
          this.isProcessing = false;
        } else if (response.data.status === "failed") {
          this.toast.error(
            `Processing failed: ${response.data.error || "Unknown error"}`
          );
          this.isProcessing = false;
        }
      } catch (error) {
        console.error("Status check error:", error);
        this.toast.error("Failed to check status");
      }
    },

    async downloadResults() {
      if (!this.jobId) return;

      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, `download/${this.jobId}`),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute("download", `patient_data_${this.jobId}.csv`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        this.toast.success("Download started!");
      } catch (error) {
        console.error("Download error:", error);
        this.toast.error("Failed to download results");
      }
    },

    resetForm() {
      this.zipFile = null;
      this.excelFile = null;
      this.pageCount = 2;
      this.jobId = null;
      this.jobStatus = null;
      this.isProcessing = false;
    },

    getStatusIcon() {
      if (!this.jobStatus) return "";

      switch (this.jobStatus.status) {
        case "completed":
          return "✅";
        case "failed":
          return "❌";
        case "processing":
          return "⏳";
        default:
          return "⏸️";
      }
    },

    // Fast processing methods
    onZipDropFast(e) {
      e.preventDefault();
      this.isZipDragActiveFast = false;
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].name.endsWith(".zip")) {
        this.zipFileFast = files[0];
        this.toast.success("Patient documents uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid ZIP archive");
      }
    },

    onExcelDropFast(e) {
      e.preventDefault();
      this.isExcelDragActiveFast = false;
      const files = e.dataTransfer.files;
      if (
        files.length > 0 &&
        (files[0].name.endsWith(".xlsx") || files[0].name.endsWith(".xls"))
      ) {
        this.excelFileFast = files[0];
        this.toast.success("Processing template uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid Excel template");
      }
    },

    triggerZipUploadFast() {
      this.$refs.zipInputFast.click();
    },

    triggerExcelUploadFast() {
      this.$refs.excelInputFast.click();
    },

    onZipFileSelectFast(e) {
      const file = e.target.files[0];
      if (file && file.name.endsWith(".zip")) {
        this.zipFileFast = file;
        this.toast.success("Patient documents uploaded successfully!");
      } else {
        this.toast.error("Please select a valid ZIP archive");
      }
    },

    onExcelFileSelectFast(e) {
      const file = e.target.files[0];
      if (file && (file.name.endsWith(".xlsx") || file.name.endsWith(".xls"))) {
        this.excelFileFast = file;
        this.toast.success("Processing template uploaded successfully!");
      } else {
        this.toast.error("Please select a valid Excel template");
      }
    },

    async startProcessingFast() {
      if (!this.canProcessFast) {
        this.toast.error("Please upload both files and set a valid page count");
        return;
      }

      this.isProcessingFast = true;
      this.jobStatusFast = null;

      const formData = new FormData();
      formData.append("zip_file", this.zipFileFast);
      formData.append("excel_file", this.excelFileFast);
      formData.append("n_pages", this.pageCountFast);
      formData.append("model", "gemini-2.5-flash"); // Use the fast model

      // Debug: Log the URL being used
      const uploadUrl = joinUrl(API_BASE_URL, "upload");
      console.log("🔧 API_BASE_URL:", API_BASE_URL);
      console.log("🔧 Upload URL:", uploadUrl);

      try {
        const response = await axios.post(uploadUrl, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        this.jobIdFast = response.data.job_id;
        this.toast.success(
          "Fast processing started! Check the status section below."
        );

        // Set initial status
        this.jobStatusFast = {
          status: "processing",
          progress: 0,
          message: "Fast processing started...",
        };
      } catch (error) {
        console.error("Upload error:", error);
        console.error("🔧 Error URL:", uploadUrl);
        this.toast.error("Failed to start fast processing. Please try again.");
        this.isProcessingFast = false;
      }
    },

    async checkJobStatusFast() {
      if (!this.jobIdFast) {
        this.toast.error("No job ID available");
        return;
      }

      try {
        const statusUrl = joinUrl(API_BASE_URL, `status/${this.jobIdFast}`);
        const response = await axios.get(statusUrl);

        this.jobStatusFast = response.data;

        if (response.data.status === "completed") {
          this.toast.success("Fast processing completed!");
          this.isProcessingFast = false;
        } else if (response.data.status === "failed") {
          this.toast.error(
            `Fast processing failed: ${response.data.error || "Unknown error"}`
          );
          this.isProcessingFast = false;
        }
      } catch (error) {
        console.error("Status check error:", error);
        this.toast.error("Failed to check status");
      }
    },

    async downloadResultsFast() {
      if (!this.jobIdFast) return;

      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, `download/${this.jobIdFast}`),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute(
          "download",
          `patient_data_fast_${this.jobIdFast}.csv`
        );
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        this.toast.success("Download started!");
      } catch (error) {
        console.error("Download error:", error);
        this.toast.error("Failed to download results");
      }
    },

    resetFormFast() {
      this.zipFileFast = null;
      this.excelFileFast = null;
      this.pageCountFast = 2;
      this.jobIdFast = null;
      this.jobStatusFast = null;
      this.isProcessingFast = false;
    },

    getStatusTitleFast() {
      if (!this.jobStatusFast) return "";

      switch (this.jobStatusFast.status) {
        case "completed":
          return "Fast Processing Complete";
        case "failed":
          return "Fast Processing Failed";
        case "processing":
          return "Fast Processing Documents";
        default:
          return "Fast Processing Status";
      }
    },

    getStatusIconFast() {
      if (!this.jobStatusFast) return "";

      switch (this.jobStatusFast.status) {
        case "completed":
          return "✅";
        case "failed":
          return "❌";
        case "processing":
          return "⚡";
        default:
          return "⏸️";
      }
    },

    // CPT prediction methods
    onCsvDrop(e) {
      e.preventDefault();
      this.isCsvDragActive = false;
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].name.endsWith(".csv")) {
        this.csvFile = files[0];
        this.toast.success("CSV file uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid CSV file");
      }
    },

    triggerCsvUpload() {
      this.$refs.csvInput.click();
    },

    onCsvFileSelect(e) {
      const file = e.target.files[0];
      if (file && file.name.endsWith(".csv")) {
        this.csvFile = file;
        this.toast.success("CSV file uploaded successfully!");
      } else {
        this.toast.error("Please select a valid CSV file");
      }
    },

    async startCptPrediction() {
      if (!this.canPredictCpt) {
        this.toast.error("Please upload a CSV file");
        return;
      }

      this.isPredictingCpt = true;
      this.cptJobStatus = null;

      const formData = new FormData();
      formData.append("csv_file", this.csvFile);
      formData.append("client", this.selectedClient);

      const uploadUrl = joinUrl(API_BASE_URL, "predict-cpt");
      console.log("🔧 CPT Upload URL:", uploadUrl);

      try {
        const response = await axios.post(uploadUrl, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        this.cptJobId = response.data.job_id;
        this.toast.success(
          "CPT prediction started! Check the status section below."
        );

        // Set initial status
        this.cptJobStatus = {
          status: "processing",
          progress: 0,
          message: "CPT prediction started...",
        };
      } catch (error) {
        console.error("CPT prediction error:", error);
        this.toast.error("Failed to start CPT prediction. Please try again.");
        this.isPredictingCpt = false;
      }
    },

    async checkCptJobStatus() {
      if (!this.cptJobId) {
        this.toast.error("No job ID available");
        return;
      }

      try {
        const statusUrl = joinUrl(API_BASE_URL, `status/${this.cptJobId}`);
        const response = await axios.get(statusUrl);

        this.cptJobStatus = response.data;

        if (response.data.status === "completed") {
          this.toast.success("CPT prediction completed!");
          this.isPredictingCpt = false;
        } else if (response.data.status === "failed") {
          this.toast.error(
            `CPT prediction failed: ${response.data.error || "Unknown error"}`
          );
          this.isPredictingCpt = false;
        }
      } catch (error) {
        console.error("Status check error:", error);
        this.toast.error("Failed to check status");
      }
    },

    async downloadCptResults() {
      if (!this.cptJobId) return;

      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, `download/${this.cptJobId}`),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute("download", `cpt_predictions_${this.cptJobId}.csv`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        this.toast.success("Download started!");
      } catch (error) {
        console.error("Download error:", error);
        this.toast.error("Failed to download results");
      }
    },

    resetCptForm() {
      this.csvFile = null;
      this.cptJobId = null;
      this.cptJobStatus = null;
      this.isPredictingCpt = false;
      this.isCsvDragActive = false;
    },

    getCptStatusTitle() {
      if (!this.cptJobStatus) return "";

      switch (this.cptJobStatus.status) {
        case "completed":
          return "CPT Prediction Complete";
        case "failed":
          return "CPT Prediction Failed";
        case "processing":
          return "Predicting CPT Codes";
        default:
          return "CPT Prediction Status";
      }
    },

    getCptStatusIcon() {
      if (!this.cptJobStatus) return "";

      switch (this.cptJobStatus.status) {
        case "completed":
          return "✅";
        case "failed":
          return "❌";
        case "processing":
          return "🏥";
        default:
          return "⏸️";
      }
    },

    // UNI conversion methods
    onUniCsvDrop(e) {
      e.preventDefault();
      this.isUniCsvDragActive = false;
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].name.endsWith(".csv")) {
        this.uniCsvFile = files[0];
        this.toast.success("UNI CSV file uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid CSV file");
      }
    },

    triggerUniCsvUpload() {
      this.$refs.uniCsvInput.click();
    },

    onUniCsvFileSelect(e) {
      const file = e.target.files[0];
      if (file && file.name.endsWith(".csv")) {
        this.uniCsvFile = file;
        this.toast.success("UNI CSV file uploaded successfully!");
      } else {
        this.toast.error("Please select a valid CSV file");
      }
    },

    async startUniConversion() {
      if (!this.canConvertUni) {
        this.toast.error("Please upload a CSV file");
        return;
      }

      this.isConvertingUni = true;
      this.uniJobStatus = null;

      const formData = new FormData();
      formData.append("csv_file", this.uniCsvFile);

      const uploadUrl = joinUrl(API_BASE_URL, "convert-uni");
      console.log("🔧 UNI Upload URL:", uploadUrl);

      try {
        const response = await axios.post(uploadUrl, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        this.uniJobId = response.data.job_id;
        this.toast.success(
          "UNI conversion started! Check the status section below."
        );

        // Set initial status
        this.uniJobStatus = {
          status: "processing",
          progress: 0,
          message: "UNI conversion started...",
        };
      } catch (error) {
        console.error("UNI conversion error:", error);
        this.toast.error("Failed to start UNI conversion. Please try again.");
        this.isConvertingUni = false;
      }
    },

    async checkUniJobStatus() {
      if (!this.uniJobId) {
        this.toast.error("No job ID available");
        return;
      }

      try {
        const statusUrl = joinUrl(API_BASE_URL, `status/${this.uniJobId}`);
        const response = await axios.get(statusUrl);

        this.uniJobStatus = response.data;

        if (response.data.status === "completed") {
          this.toast.success("UNI conversion completed!");
          this.isConvertingUni = false;
        } else if (response.data.status === "failed") {
          this.toast.error(
            `UNI conversion failed: ${response.data.error || "Unknown error"}`
          );
          this.isConvertingUni = false;
        }
      } catch (error) {
        console.error("Status check error:", error);
        this.toast.error("Failed to check status");
      }
    },

    async downloadUniResults() {
      if (!this.uniJobId) return;

      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, `download/${this.uniJobId}`),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute("download", `uni_converted_${this.uniJobId}.csv`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        this.toast.success("Download started!");
      } catch (error) {
        console.error("Download error:", error);
        this.toast.error("Failed to download results");
      }
    },

    resetUniForm() {
      this.uniCsvFile = null;
      this.uniJobId = null;
      this.uniJobStatus = null;
      this.isConvertingUni = false;
      this.isUniCsvDragActive = false;
    },

    getUniStatusTitle() {
      if (!this.uniJobStatus) return "";

      switch (this.uniJobStatus.status) {
        case "completed":
          return "UNI Conversion Complete";
        case "failed":
          return "UNI Conversion Failed";
        case "processing":
          return "Converting UNI CSV";
        default:
          return "UNI Conversion Status";
      }
    },

    getUniStatusIcon() {
      if (!this.uniJobStatus) return "";

      switch (this.uniJobStatus.status) {
        case "completed":
          return "✅";
        case "failed":
          return "❌";
        case "processing":
          return "🔄";
        default:
          return "⏸️";
      }
    },
  },
};
</script>

<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap");

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    sans-serif;
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

.tab-btn.disabled {
  background: #f8fafc;
  border-color: #e2e8f0;
  color: #94a3b8;
  cursor: not-allowed;
  opacity: 0.6;
}

.tab-btn.disabled:hover {
  transform: none;
  border-color: #e2e8f0;
  color: #94a3b8;
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

/* Form Elements */
.form-group {
  margin-bottom: 1rem;
}

.form-label {
  display: block;
  font-weight: 500;
  color: #374151;
  margin-bottom: 0.5rem;
  font-size: 0.875rem;
}

.form-select {
  width: 100%;
  padding: 0.75rem 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 500;
  transition: all 0.3s ease;
  background: white;
  cursor: pointer;
}

.form-select:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.form-select:hover {
  border-color: #cbd5e1;
}

/* Action Section */
.action-section {
  display: flex;
  gap: 1rem;
  justify-content: center;
  align-items: center;
  flex-wrap: wrap;
}

.process-btn,
.reset-btn {
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
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
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

.check-status-btn {
  background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 1rem;
  box-shadow: 0 4px 6px -1px rgba(245, 158, 11, 0.3);
}

.check-status-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 12px -2px rgba(245, 158, 11, 0.4);
}

.processing-note {
  margin-top: 1rem;
  font-size: 0.875rem;
  color: #64748b;
  font-style: italic;
  text-align: center;
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

/* Requirement List Styles */
.requirement-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.requirement-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  background: #f8fafc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}

.requirement-icon {
  font-size: 1.25rem;
  flex-shrink: 0;
}

.requirement-item span:last-child {
  color: #374151;
  font-size: 0.875rem;
  font-weight: 500;
}

/* Disabled Section Styles */
.disabled-section {
  display: flex;
  justify-content: center;
  margin: 2rem 0;
}

.disabled-card {
  background: #f8fafc;
  border: 2px solid #e2e8f0;
  border-radius: 16px;
  padding: 3rem;
  text-align: center;
  max-width: 500px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

.disabled-icon {
  font-size: 4rem;
  margin-bottom: 1.5rem;
  opacity: 0.7;
}

.disabled-card h3 {
  font-size: 1.5rem;
  font-weight: 600;
  color: #64748b;
  margin-bottom: 1rem;
}

.disabled-card p {
  color: #64748b;
  font-size: 1rem;
  line-height: 1.6;
  margin-bottom: 2rem;
}

.disabled-features {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.feature-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem;
  background: white;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
  justify-content: center;
}

.feature-icon {
  font-size: 1.25rem;
  flex-shrink: 0;
}

.feature-item span:last-child {
  color: #374151;
  font-size: 0.875rem;
  font-weight: 500;
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

  .process-btn,
  .reset-btn {
    width: 100%;
  }

  .status-header {
    flex-direction: column;
    text-align: center;
    gap: 1rem;
  }
}
</style>
