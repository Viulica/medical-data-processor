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
          <button
            @click="activeTab = 'instructions'"
            :class="{ active: activeTab === 'instructions' }"
            class="tab-btn"
          >
            📋 Convert Instructions
          </button>
          <button
            @click="activeTab = 'modifiers'"
            :class="{ active: activeTab === 'modifiers' }"
            class="tab-btn"
          >
            💊 Generate Modifiers
          </button>
          <button
            @click="activeTab = 'insurance'"
            :class="{ active: activeTab === 'insurance' }"
            class="tab-btn"
          >
            🏥 Insurance Sorting
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
              Upload a CSV file with a 'Procedure Description' column to predict
              anesthesia CPT codes using AI
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
                    <option value="tan-esc">TAN-ESC (Custom Model)</option>
                  </select>
                </div>
              </div>
            </div>

            <!-- Custom Instructions -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">3</div>
                <h3>Custom Coding Instructions</h3>
              </div>
              <div class="settings-content">
                <div class="form-group">
                  <label for="custom-instructions" class="form-label">
                    Additional instructions for medical coder (optional):
                  </label>
                  <textarea
                    id="custom-instructions"
                    v-model="customInstructions"
                    class="form-textarea"
                    rows="4"
                    placeholder="Enter specific coding guidelines, corrections, or preferences that should override AI predictions..."
                  ></textarea>
                  <div class="instruction-actions">
                    <button
                      @click="saveInstructions"
                      class="save-btn"
                      :disabled="isSavingInstructions"
                    >
                      <span v-if="isSavingInstructions" class="spinner"></span>
                      <span v-else>💾</span>
                      {{
                        isSavingInstructions ? "Saving..." : "Save Instructions"
                      }}
                    </button>
                    <button
                      @click="loadInstructions"
                      class="load-btn"
                      :disabled="isLoadingInstructions"
                    >
                      <span v-if="isLoadingInstructions" class="spinner"></span>
                      <span v-else>📂</span>
                      {{ isLoadingInstructions ? "Loading..." : "Load Saved" }}
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <!-- Requirements -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">4</div>
                <h3>Requirements</h3>
              </div>
              <div class="settings-content">
                <div class="requirement-list">
                  <div class="requirement-item">
                    <span class="requirement-icon">📋</span>
                    <span
                      >CSV must contain a 'Procedure Description' column</span
                    >
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">🏥</span>
                    <span>Each row should have a procedure description</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">⚡</span>
                    <span
                      >AI will predict CPT codes in 'ASA Code' and 'Procedure
                      Code' columns</span
                    >
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

        <!-- Instructions Conversion Tab -->
        <div v-if="activeTab === 'instructions'" class="upload-section">
          <div class="section-header">
            <h2>Instructions Conversion</h2>
            <p>
              Upload an Excel file to convert it using the automated
              instructions conversion script
            </p>
          </div>

          <div class="upload-grid">
            <!-- Excel File Upload -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">1</div>
                <h3>Instructions Excel File</h3>
              </div>
              <div
                class="dropzone"
                :class="{
                  active: isInstructionsExcelDragActive,
                  'has-file': instructionsExcelFile,
                }"
                @drop="onInstructionsExcelDrop"
                @dragover.prevent
                @dragenter.prevent
                @click="triggerInstructionsExcelUpload"
              >
                <input
                  ref="instructionsExcelInput"
                  type="file"
                  accept=".xlsx,.xls"
                  @change="onInstructionsExcelFileSelect"
                  style="display: none"
                />
                <div class="upload-content">
                  <div class="upload-icon">📊</div>
                  <div v-if="instructionsExcelFile" class="file-info">
                    <div class="file-icon">📄</div>
                    <span class="file-name">{{
                      instructionsExcelFile.name
                    }}</span>
                    <span class="file-size">{{
                      formatFileSize(instructionsExcelFile.size)
                    }}</span>
                  </div>
                  <p v-else class="upload-text">
                    Drag & drop Excel file here<br />or click to browse
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
                    <span>Excel file with GAP instruction format</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">🔄</span>
                    <span>Automatic conversion using mapping rules</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">📥</span>
                    <span>Download converted Excel file</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Action Buttons -->
          <div class="action-section">
            <button
              @click="startInstructionsConversion"
              :disabled="!canConvertInstructions || isConvertingInstructions"
              class="process-btn"
            >
              <span v-if="isConvertingInstructions" class="spinner"></span>
              <span v-else class="btn-icon">📋</span>
              {{
                isConvertingInstructions
                  ? "Converting Instructions..."
                  : "Start Instructions Conversion"
              }}
            </button>

            <button
              v-if="instructionsExcelFile || instructionsJobId"
              @click="resetInstructionsForm"
              class="reset-btn"
            >
              Reset
            </button>
          </div>
        </div>

        <!-- Instructions Conversion Status -->
        <div v-if="instructionsJobStatus" class="status-section">
          <div class="status-card">
            <div class="status-header">
              <div
                class="status-indicator"
                :class="instructionsJobStatus.status"
              >
                <span class="status-icon">{{
                  getInstructionsStatusIcon()
                }}</span>
              </div>
              <div class="status-info">
                <h3>{{ getInstructionsStatusTitle() }}</h3>
                <p class="status-message">
                  {{ instructionsJobStatus.message }}
                </p>
              </div>
            </div>

            <div
              v-if="instructionsJobStatus.status === 'processing'"
              class="progress-section"
            >
              <div class="progress-bar">
                <div
                  class="progress-fill"
                  :style="{ width: `${instructionsJobStatus.progress}%` }"
                ></div>
              </div>
              <div class="progress-text">
                {{ instructionsJobStatus.progress }}% Complete
              </div>
              <button
                @click="checkInstructionsJobStatus"
                class="check-status-btn"
              >
                <span class="btn-icon">🔄</span>
                Check Status
              </button>
            </div>

            <div
              v-if="instructionsJobStatus.status === 'completed'"
              class="success-section"
            >
              <button @click="downloadInstructionsResults" class="download-btn">
                <span class="btn-icon">📥</span>
                Download Converted Excel
              </button>
            </div>

            <div
              v-if="
                instructionsJobStatus.status === 'failed' &&
                instructionsJobStatus.error
              "
              class="error-section"
            >
              <div class="error-message">
                <span class="error-icon">⚠️</span>
                <span>{{ instructionsJobStatus.error }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Modifiers Generation Tab -->
        <div v-if="activeTab === 'modifiers'" class="upload-section">
          <div class="section-header">
            <h2>Medical Modifiers Generation</h2>
            <p>
              Upload a CSV file with billing data to automatically generate
              medical modifiers based on provider information
            </p>
          </div>

          <div class="upload-grid">
            <!-- CSV File Upload -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">1</div>
                <h3>Billing CSV File</h3>
              </div>
              <div
                class="dropzone"
                :class="{
                  active: isModifiersCsvDragActive,
                  'has-file': modifiersCsvFile,
                }"
                @drop="onModifiersCsvDrop"
                @dragover.prevent
                @dragenter.prevent
                @click="triggerModifiersCsvUpload"
              >
                <input
                  ref="modifiersCsvInput"
                  type="file"
                  accept=".csv"
                  @change="onModifiersCsvFileSelect"
                  style="display: none"
                />
                <div class="upload-content">
                  <div class="upload-icon">📊</div>
                  <div v-if="modifiersCsvFile" class="file-info">
                    <div class="file-icon">📄</div>
                    <span class="file-name">{{ modifiersCsvFile.name }}</span>
                    <span class="file-size">{{
                      formatFileSize(modifiersCsvFile.size)
                    }}</span>
                  </div>
                  <p v-else class="upload-text">
                    Drag & drop CSV file here<br />or click to browse
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
                    <span
                      >CSV must contain 'Primary Mednet Code', 'MD', and 'CRNA'
                      columns</span
                    >
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">💊</span>
                    <span
                      >Modifiers will be generated in M1 column based on
                      provider types</span
                    >
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">📥</span>
                    <span>Download CSV with modifiers applied</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">🔀</span>
                    <span
                      >Rows may be duplicated for QK+QX modifier
                      combinations</span
                    >
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Action Buttons -->
          <div class="action-section">
            <button
              @click="startModifiersGeneration"
              :disabled="!canGenerateModifiers || isGeneratingModifiers"
              class="process-btn"
            >
              <span v-if="isGeneratingModifiers" class="spinner"></span>
              <span v-else class="btn-icon">💊</span>
              {{
                isGeneratingModifiers
                  ? "Generating Modifiers..."
                  : "Generate Modifiers"
              }}
            </button>

            <button
              v-if="modifiersCsvFile || modifiersJobId"
              @click="resetModifiersForm"
              class="reset-btn"
            >
              Reset
            </button>
          </div>
        </div>

        <!-- Modifiers Generation Status -->
        <div v-if="modifiersJobStatus" class="status-section">
          <div class="status-card">
            <div class="status-header">
              <div class="status-indicator" :class="modifiersJobStatus.status">
                <span class="status-icon">{{ getModifiersStatusIcon() }}</span>
              </div>
              <div class="status-info">
                <h3>{{ getModifiersStatusTitle() }}</h3>
                <p class="status-message">
                  {{ modifiersJobStatus.message }}
                </p>
              </div>
            </div>

            <div
              v-if="modifiersJobStatus.status === 'processing'"
              class="progress-section"
            >
              <div class="progress-bar">
                <div
                  class="progress-fill"
                  :style="{ width: `${modifiersJobStatus.progress}%` }"
                ></div>
              </div>
              <div class="progress-text">
                {{ modifiersJobStatus.progress }}% Complete
              </div>
              <button @click="checkModifiersJobStatus" class="check-status-btn">
                <span class="btn-icon">🔄</span>
                Check Status
              </button>
            </div>

            <div
              v-if="modifiersJobStatus.status === 'completed'"
              class="success-section"
            >
              <button @click="downloadModifiersResults" class="download-btn">
                <span class="btn-icon">📥</span>
                Download CSV with Modifiers
              </button>
            </div>

            <div
              v-if="
                modifiersJobStatus.status === 'failed' &&
                modifiersJobStatus.error
              "
              class="error-section"
            >
              <div class="error-message">
                <span class="error-icon">⚠️</span>
                <span>{{ modifiersJobStatus.error }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Insurance Sorting Tab -->
        <div v-if="activeTab === 'insurance'" class="upload-section">
          <div class="section-header">
            <h2>Insurance MedNet Code Prediction</h2>
            <p>
              Upload patient insurance data to automatically predict MedNet
              codes for primary, secondary, and tertiary insurance using AI
            </p>
          </div>

          <div class="upload-grid">
            <!-- Data CSV Upload -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">1</div>
                <h3>Patient Insurance Data (Required)</h3>
              </div>
              <div
                class="dropzone"
                :class="{
                  active: isInsuranceDataDragActive,
                  'has-file': insuranceDataCsv,
                }"
                @drop="onInsuranceDataCsvDrop"
                @dragover.prevent
                @dragenter.prevent
                @click="triggerInsuranceDataCsvUpload"
              >
                <input
                  ref="insuranceDataCsvInput"
                  type="file"
                  accept=".csv"
                  @change="onInsuranceDataCsvFileSelect"
                  style="display: none"
                />
                <div class="upload-content">
                  <div class="upload-icon">📊</div>
                  <div v-if="insuranceDataCsv" class="file-info">
                    <div class="file-icon">📄</div>
                    <span class="file-name">{{ insuranceDataCsv.name }}</span>
                    <span class="file-size">{{
                      formatFileSize(insuranceDataCsv.size)
                    }}</span>
                  </div>
                  <p v-else class="upload-text">
                    Drag & drop insurance data CSV<br />or click to browse
                  </p>
                </div>
              </div>
              <div class="field-info">
                <p class="info-text">
                  ℹ️ CSV must contain: <strong>Primary Company Name</strong>,
                  <strong>Primary Company Address 1</strong> (and optionally
                  Secondary/Tertiary equivalents)
                </p>
              </div>
            </div>

            <!-- Special Cases CSV Upload (Optional) -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">2</div>
                <h3>Special Cases (Optional)</h3>
              </div>
              <div
                class="dropzone"
                :class="{
                  active: isSpecialCasesDragActive,
                  'has-file': specialCasesCsv,
                }"
                @drop="onSpecialCasesCsvDrop"
                @dragover.prevent
                @dragenter.prevent
                @click="triggerSpecialCasesCsvUpload"
              >
                <input
                  ref="specialCasesCsvInput"
                  type="file"
                  accept=".csv"
                  @change="onSpecialCasesCsvFileSelect"
                  style="display: none"
                />
                <div class="upload-content">
                  <div class="upload-icon">📋</div>
                  <div v-if="specialCasesCsv" class="file-info">
                    <div class="file-icon">📄</div>
                    <span class="file-name">{{ specialCasesCsv.name }}</span>
                    <span class="file-size">{{
                      formatFileSize(specialCasesCsv.size)
                    }}</span>
                  </div>
                  <p v-else class="upload-text">
                    Optional: Upload special cases<br />or skip to use defaults
                  </p>
                </div>
              </div>
              <div class="field-info">
                <p class="info-text">
                  ℹ️ Optional CSV with custom mappings. Must contain:
                  <strong>Company name</strong>, <strong>Mednet code</strong>
                </p>
              </div>
            </div>
          </div>

          <!-- Requirements Info -->
          <div class="upload-grid">
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">3</div>
                <h3>AI Settings</h3>
              </div>
              <div class="settings-content">
                <div class="ai-toggle-container">
                  <div class="toggle-header">
                    <label class="toggle-label">
                      <span class="toggle-icon">🤖</span>
                      <span class="toggle-text">Enable AI Matching</span>
                    </label>
                    <label class="switch">
                      <input type="checkbox" v-model="enableAi" />
                      <span class="slider"></span>
                    </label>
                  </div>
                  <p class="toggle-description">
                    {{
                      enableAi
                        ? "AI will analyze insurance names and addresses to predict MedNet codes with high confidence matching."
                        : "Only special cases will be mapped. Regular PO Box matching is disabled."
                    }}
                  </p>
                </div>
                <div class="requirement-list" style="margin-top: 20px">
                  <div class="requirement-item">
                    <span class="requirement-icon">🔍</span>
                    <span>Extracts PO Box from insurance addresses</span>
                  </div>
                  <div class="requirement-item">
                    <span
                      class="requirement-icon"
                      :style="{ opacity: enableAi ? 1 : 0.3 }"
                      >🤖</span
                    >
                    <span :style="{ opacity: enableAi ? 1 : 0.5 }"
                      >AI matches insurance names to MedNet database</span
                    >
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">📋</span>
                    <span
                      >Predicts codes for Primary, Secondary, and Tertiary
                      insurance</span
                    >
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">✨</span>
                    <span>Special cases override automatic predictions</span>
                  </div>
                </div>
              </div>
            </div>

            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">4</div>
                <h3>Output</h3>
              </div>
              <div class="settings-content">
                <div class="requirement-list">
                  <div class="requirement-item">
                    <span class="requirement-icon">📊</span>
                    <span>Original CSV with added MedNet code columns</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">🎯</span>
                    <span>Columns: Primary/Secondary/Tertiary Mednet Code</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">📥</span>
                    <span>Download enriched CSV with predictions</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Action Buttons -->
          <div class="action-section">
            <button
              @click="startInsurancePrediction"
              :disabled="!canPredictInsurance || isPredictingInsurance"
              class="process-btn"
            >
              <span v-if="isPredictingInsurance" class="spinner"></span>
              <span v-else class="btn-icon">🏥</span>
              {{
                isPredictingInsurance
                  ? "Predicting Insurance Codes..."
                  : "Start Insurance Prediction"
              }}
            </button>

            <button
              v-if="insuranceDataCsv || insuranceJobId"
              @click="resetInsuranceForm"
              class="reset-btn"
            >
              Reset
            </button>
          </div>
        </div>

        <!-- Insurance Sorting Status -->
        <div v-if="insuranceJobStatus" class="status-section">
          <div class="status-card">
            <div class="status-header">
              <div class="status-indicator" :class="insuranceJobStatus.status">
                <span class="status-icon">{{ getInsuranceStatusIcon() }}</span>
              </div>
              <div class="status-info">
                <h3>{{ getInsuranceStatusTitle() }}</h3>
                <p class="status-message">
                  {{ insuranceJobStatus.message }}
                </p>
              </div>
            </div>

            <div
              v-if="insuranceJobStatus.status === 'processing'"
              class="progress-section"
            >
              <div class="progress-bar">
                <div
                  class="progress-fill"
                  :style="{ width: `${insuranceJobStatus.progress}%` }"
                ></div>
              </div>
              <div class="progress-text">
                {{ insuranceJobStatus.progress }}% Complete
              </div>
              <button @click="checkInsuranceJobStatus" class="check-status-btn">
                <span class="btn-icon">🔄</span>
                Check Status
              </button>
            </div>

            <div
              v-if="insuranceJobStatus.status === 'completed'"
              class="success-section"
            >
              <button @click="downloadInsuranceResults" class="download-btn">
                <span class="btn-icon">📥</span>
                Download CSV with Insurance Codes
              </button>
            </div>

            <div
              v-if="
                insuranceJobStatus.status === 'failed' &&
                insuranceJobStatus.error
              "
              class="error-section"
            >
              <div class="error-message">
                <span class="error-icon">⚠️</span>
                <span>{{ insuranceJobStatus.error }}</span>
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
      customInstructions: "",
      isSavingInstructions: false,
      isLoadingInstructions: false,
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
      // Instructions conversion functionality
      instructionsExcelFile: null,
      instructionsJobId: null,
      instructionsJobStatus: null,
      isConvertingInstructions: false,
      isInstructionsExcelDragActive: false,
      // Modifiers generation functionality
      modifiersCsvFile: null,
      modifiersJobId: null,
      modifiersJobStatus: null,
      isGeneratingModifiers: false,
      isModifiersCsvDragActive: false,
      // Insurance sorting functionality
      insuranceDataCsv: null,
      specialCasesCsv: null,
      insuranceJobId: null,
      insuranceJobStatus: null,
      isPredictingInsurance: false,
      isInsuranceDataDragActive: false,
      isSpecialCasesDragActive: false,
      enableAi: true, // Toggle for AI-powered insurance matching
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
    canConvertInstructions() {
      return this.instructionsExcelFile;
    },
    canGenerateModifiers() {
      return this.modifiersCsvFile;
    },
    canPredictInsurance() {
      return this.insuranceDataCsv;
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
      formData.append("model", "gemini-flash-latest"); // Use the fast model

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

      // Use custom endpoint for tan-esc model
      let uploadUrl;
      if (this.selectedClient === "tan-esc") {
        uploadUrl = joinUrl(API_BASE_URL, "predict-cpt-custom");
        formData.append("confidence_threshold", "0.5");
        console.log("🔧 CPT Upload URL (Custom Model):", uploadUrl);
      } else {
        uploadUrl = joinUrl(API_BASE_URL, "predict-cpt");
        formData.append("client", this.selectedClient);
        console.log("🔧 CPT Upload URL:", uploadUrl);
      }

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

    async saveInstructions() {
      if (!API_BASE_URL) {
        this.toast.error("API not available");
        return;
      }

      this.isSavingInstructions = true;
      try {
        await axios.post(
          joinUrl(API_BASE_URL, "save-instructions"),
          { instructions: this.customInstructions || "" },
          { headers: { "Content-Type": "application/json" } }
        );
        this.toast.success("Instructions saved successfully!");
      } catch (error) {
        console.error("Save instructions error:", error);
        this.toast.error("Failed to save instructions");
      } finally {
        this.isSavingInstructions = false;
      }
    },

    async loadInstructions() {
      if (!API_BASE_URL) {
        console.warn("API_BASE_URL not available, skipping instruction load");
        return;
      }

      this.isLoadingInstructions = true;
      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, "load-instructions")
        );
        this.customInstructions = response.data?.instructions || "";
        this.toast.success("Instructions loaded successfully!");
      } catch (error) {
        console.error("Load instructions error:", error);
        if (error.response?.status === 404) {
          this.customInstructions = "";
          this.toast.info("No saved instructions found");
        } else {
          this.toast.error("Failed to load instructions");
        }
      } finally {
        this.isLoadingInstructions = false;
      }
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

    // Instructions conversion methods
    onInstructionsExcelDrop(e) {
      e.preventDefault();
      this.isInstructionsExcelDragActive = false;
      const files = e.dataTransfer.files;
      if (
        files.length > 0 &&
        (files[0].name.endsWith(".xlsx") || files[0].name.endsWith(".xls"))
      ) {
        this.instructionsExcelFile = files[0];
        this.toast.success("Instructions Excel file uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid Excel file");
      }
    },

    triggerInstructionsExcelUpload() {
      this.$refs.instructionsExcelInput.click();
    },

    onInstructionsExcelFileSelect(e) {
      const file = e.target.files[0];
      if (file && (file.name.endsWith(".xlsx") || file.name.endsWith(".xls"))) {
        this.instructionsExcelFile = file;
        this.toast.success("Instructions Excel file uploaded successfully!");
      } else {
        this.toast.error("Please select a valid Excel file");
      }
    },

    async startInstructionsConversion() {
      if (!this.canConvertInstructions) {
        this.toast.error("Please upload an Excel file");
        return;
      }

      this.isConvertingInstructions = true;
      this.instructionsJobStatus = null;

      const formData = new FormData();
      formData.append("excel_file", this.instructionsExcelFile);

      const uploadUrl = joinUrl(API_BASE_URL, "convert-instructions");
      console.log("🔧 Instructions Upload URL:", uploadUrl);

      try {
        const response = await axios.post(uploadUrl, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        this.instructionsJobId = response.data.job_id;
        this.toast.success(
          "Instructions conversion started! Check the status section below."
        );

        // Set initial status
        this.instructionsJobStatus = {
          status: "processing",
          progress: 0,
          message: "Instructions conversion started...",
        };
      } catch (error) {
        console.error("Instructions conversion error:", error);
        this.toast.error(
          "Failed to start instructions conversion. Please try again."
        );
        this.isConvertingInstructions = false;
      }
    },

    async checkInstructionsJobStatus() {
      if (!this.instructionsJobId) {
        this.toast.error("No job ID available");
        return;
      }

      try {
        const statusUrl = joinUrl(
          API_BASE_URL,
          `status/${this.instructionsJobId}`
        );
        const response = await axios.get(statusUrl);

        this.instructionsJobStatus = response.data;

        if (response.data.status === "completed") {
          this.toast.success("Instructions conversion completed!");
          this.isConvertingInstructions = false;
        } else if (response.data.status === "failed") {
          this.toast.error(
            `Instructions conversion failed: ${
              response.data.error || "Unknown error"
            }`
          );
          this.isConvertingInstructions = false;
        }
      } catch (error) {
        console.error("Status check error:", error);
        this.toast.error("Failed to check status");
      }
    },

    async downloadInstructionsResults() {
      if (!this.instructionsJobId) return;

      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, `download/${this.instructionsJobId}`),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute(
          "download",
          `instructions_converted_${this.instructionsJobId}.xlsx`
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

    resetInstructionsForm() {
      this.instructionsExcelFile = null;
      this.instructionsJobId = null;
      this.instructionsJobStatus = null;
      this.isConvertingInstructions = false;
      this.isInstructionsExcelDragActive = false;
    },

    getInstructionsStatusTitle() {
      if (!this.instructionsJobStatus) return "";

      switch (this.instructionsJobStatus.status) {
        case "completed":
          return "Instructions Conversion Complete";
        case "failed":
          return "Instructions Conversion Failed";
        case "processing":
          return "Converting Instructions";
        default:
          return "Instructions Conversion Status";
      }
    },

    getInstructionsStatusIcon() {
      if (!this.instructionsJobStatus) return "";

      switch (this.instructionsJobStatus.status) {
        case "completed":
          return "✅";
        case "failed":
          return "❌";
        case "processing":
          return "📋";
        default:
          return "⏸️";
      }
    },

    // Modifiers generation methods
    onModifiersCsvDrop(e) {
      e.preventDefault();
      this.isModifiersCsvDragActive = false;
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].name.endsWith(".csv")) {
        this.modifiersCsvFile = files[0];
        this.toast.success("Modifiers CSV file uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid CSV file");
      }
    },

    triggerModifiersCsvUpload() {
      this.$refs.modifiersCsvInput.click();
    },

    onModifiersCsvFileSelect(e) {
      const file = e.target.files[0];
      if (file && file.name.endsWith(".csv")) {
        this.modifiersCsvFile = file;
        this.toast.success("Modifiers CSV file uploaded successfully!");
      } else {
        this.toast.error("Please select a valid CSV file");
      }
    },

    async startModifiersGeneration() {
      if (!this.canGenerateModifiers) {
        this.toast.error("Please upload a CSV file");
        return;
      }

      this.isGeneratingModifiers = true;
      this.modifiersJobStatus = null;

      const formData = new FormData();
      formData.append("csv_file", this.modifiersCsvFile);

      const uploadUrl = joinUrl(API_BASE_URL, "generate-modifiers");
      console.log("🔧 Modifiers Upload URL:", uploadUrl);

      try {
        const response = await axios.post(uploadUrl, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        this.modifiersJobId = response.data.job_id;
        this.toast.success(
          "Modifiers generation started! Check the status section below."
        );

        // Set initial status
        this.modifiersJobStatus = {
          status: "processing",
          progress: 0,
          message: "Modifiers generation started...",
        };
      } catch (error) {
        console.error("Modifiers generation error:", error);
        this.toast.error(
          "Failed to start modifiers generation. Please try again."
        );
        this.isGeneratingModifiers = false;
      }
    },

    async checkModifiersJobStatus() {
      if (!this.modifiersJobId) {
        this.toast.error("No job ID available");
        return;
      }

      try {
        const statusUrl = joinUrl(
          API_BASE_URL,
          `status/${this.modifiersJobId}`
        );
        const response = await axios.get(statusUrl);

        this.modifiersJobStatus = response.data;

        if (response.data.status === "completed") {
          this.toast.success("Modifiers generation completed!");
          this.isGeneratingModifiers = false;
        } else if (response.data.status === "failed") {
          this.toast.error(
            `Modifiers generation failed: ${
              response.data.error || "Unknown error"
            }`
          );
          this.isGeneratingModifiers = false;
        }
      } catch (error) {
        console.error("Status check error:", error);
        this.toast.error("Failed to check status");
      }
    },

    async downloadModifiersResults() {
      if (!this.modifiersJobId) return;

      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, `download/${this.modifiersJobId}`),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute(
          "download",
          `modifiers_generated_${this.modifiersJobId}.csv`
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

    resetModifiersForm() {
      this.modifiersCsvFile = null;
      this.modifiersJobId = null;
      this.modifiersJobStatus = null;
      this.isGeneratingModifiers = false;
      this.isModifiersCsvDragActive = false;
    },

    getModifiersStatusTitle() {
      if (!this.modifiersJobStatus) return "";

      switch (this.modifiersJobStatus.status) {
        case "completed":
          return "Modifiers Generation Complete";
        case "failed":
          return "Modifiers Generation Failed";
        case "processing":
          return "Generating Modifiers";
        default:
          return "Modifiers Generation Status";
      }
    },

    getModifiersStatusIcon() {
      if (!this.modifiersJobStatus) return "";

      switch (this.modifiersJobStatus.status) {
        case "completed":
          return "✅";
        case "failed":
          return "❌";
        case "processing":
          return "💊";
        default:
          return "⏸️";
      }
    },

    // Insurance Sorting Methods
    onInsuranceDataCsvDrop(e) {
      e.preventDefault();
      this.isInsuranceDataDragActive = false;
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].name.endsWith(".csv")) {
        this.insuranceDataCsv = files[0];
        this.toast.success("Insurance data CSV uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid CSV file");
      }
    },

    onInsuranceDataCsvFileSelect(e) {
      const file = e.target.files[0];
      if (file && file.name.endsWith(".csv")) {
        this.insuranceDataCsv = file;
        this.toast.success("Insurance data CSV uploaded successfully!");
      } else {
        this.toast.error("Please select a valid CSV file");
      }
    },

    triggerInsuranceDataCsvUpload() {
      this.$refs.insuranceDataCsvInput.click();
    },

    onSpecialCasesCsvDrop(e) {
      e.preventDefault();
      this.isSpecialCasesDragActive = false;
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].name.endsWith(".csv")) {
        this.specialCasesCsv = files[0];
        this.toast.success("Special cases CSV uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid CSV file");
      }
    },

    onSpecialCasesCsvFileSelect(e) {
      const file = e.target.files[0];
      if (file && file.name.endsWith(".csv")) {
        this.specialCasesCsv = file;
        this.toast.success("Special cases CSV uploaded successfully!");
      } else {
        this.toast.error("Please select a valid CSV file");
      }
    },

    triggerSpecialCasesCsvUpload() {
      this.$refs.specialCasesCsvInput.click();
    },

    async startInsurancePrediction() {
      if (!this.canPredictInsurance) {
        this.toast.error("Please upload insurance data CSV");
        return;
      }

      this.isPredictingInsurance = true;
      this.insuranceJobStatus = null;

      const formData = new FormData();
      formData.append("data_csv", this.insuranceDataCsv);
      if (this.specialCasesCsv) {
        formData.append("special_cases_csv", this.specialCasesCsv);
      }
      formData.append("enable_ai", this.enableAi);

      const predictUrl = joinUrl(API_BASE_URL, "predict-insurance-codes");
      console.log("🔧 Insurance Prediction URL:", predictUrl);
      console.log("🤖 AI Enabled:", this.enableAi);

      try {
        const response = await axios.post(predictUrl, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          timeout: 600000, // 10 minute timeout
        });

        this.insuranceJobId = response.data.job_id;
        this.toast.success(
          `Insurance code prediction started${
            this.enableAi ? " with AI" : " (special cases only)"
          }! Check the status section below.`
        );

        // Set initial status
        this.insuranceJobStatus = {
          status: "processing",
          progress: 0,
          message: "Processing started...",
        };

        // Start checking status automatically
        this.startInsuranceStatusPolling();
      } catch (error) {
        console.error("Insurance prediction error:", error);
        let errorMessage =
          "Failed to start insurance code prediction. Please try again.";
        if (error.response?.data?.detail) {
          errorMessage = `Server error: ${error.response.data.detail}`;
        }
        this.toast.error(errorMessage);
        this.isPredictingInsurance = false;
      }
    },

    startInsuranceStatusPolling() {
      // Check status every 5 seconds
      const pollInterval = setInterval(async () => {
        if (!this.insuranceJobId) {
          clearInterval(pollInterval);
          return;
        }

        try {
          const statusUrl = joinUrl(
            API_BASE_URL,
            `status/${this.insuranceJobId}`
          );
          const response = await axios.get(statusUrl);

          this.insuranceJobStatus = response.data;

          if (response.data.status === "completed") {
            clearInterval(pollInterval);
            this.toast.success("Insurance code prediction completed!");
            this.isPredictingInsurance = false;
          } else if (response.data.status === "failed") {
            clearInterval(pollInterval);
            this.toast.error(
              `Insurance code prediction failed: ${
                response.data.error || "Unknown error"
              }`
            );
            this.isPredictingInsurance = false;
          }
        } catch (error) {
          console.error("Status check error:", error);
        }
      }, 5000); // Check every 5 seconds
    },

    async checkInsuranceJobStatus() {
      if (!this.insuranceJobId) {
        this.toast.error("No job ID available");
        return;
      }

      try {
        const statusUrl = joinUrl(
          API_BASE_URL,
          `status/${this.insuranceJobId}`
        );
        const response = await axios.get(statusUrl);

        this.insuranceJobStatus = response.data;

        if (response.data.status === "completed") {
          this.toast.success("Insurance code prediction completed!");
          this.isPredictingInsurance = false;
        } else if (response.data.status === "failed") {
          this.toast.error(
            `Insurance code prediction failed: ${
              response.data.error || "Unknown error"
            }`
          );
          this.isPredictingInsurance = false;
        }
      } catch (error) {
        console.error("Status check error:", error);
        this.toast.error("Failed to check status");
      }
    },

    async downloadInsuranceResults() {
      if (!this.insuranceJobId) return;

      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, `download/${this.insuranceJobId}`),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute(
          "download",
          `insurance_codes_${this.insuranceJobId}.csv`
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

    resetInsuranceForm() {
      this.insuranceDataCsv = null;
      this.specialCasesCsv = null;
      this.insuranceJobId = null;
      this.insuranceJobStatus = null;
      this.isPredictingInsurance = false;
      this.isInsuranceDataDragActive = false;
      this.isSpecialCasesDragActive = false;

      // Reset file inputs
      if (this.$refs.insuranceDataCsvInput) {
        this.$refs.insuranceDataCsvInput.value = "";
      }
      if (this.$refs.specialCasesCsvInput) {
        this.$refs.specialCasesCsvInput.value = "";
      }
    },

    getInsuranceStatusTitle() {
      if (!this.insuranceJobStatus) return "";

      switch (this.insuranceJobStatus.status) {
        case "completed":
          return "Insurance Code Prediction Complete";
        case "failed":
          return "Insurance Code Prediction Failed";
        case "processing":
          return "Predicting Insurance Codes";
        default:
          return "Insurance Code Prediction Status";
      }
    },

    getInsuranceStatusIcon() {
      if (!this.insuranceJobStatus) return "";

      switch (this.insuranceJobStatus.status) {
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

.form-textarea {
  width: 100%;
  padding: 0.75rem 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 0.875rem;
  font-family: inherit;
  line-height: 1.5;
  transition: all 0.3s ease;
  background: white;
  resize: vertical;
  min-height: 100px;
}

.form-textarea:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.form-textarea:hover {
  border-color: #cbd5e1;
}

.instruction-actions {
  display: flex;
  gap: 0.75rem;
  margin-top: 0.75rem;
}

.save-btn,
.load-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
}

.save-btn {
  background: #10b981;
  color: white;
}

.save-btn:hover:not(:disabled) {
  background: #059669;
}

.save-btn:disabled {
  background: #9ca3af;
  cursor: not-allowed;
}

.load-btn {
  background: #3b82f6;
  color: white;
}

.load-btn:hover:not(:disabled) {
  background: #2563eb;
}

.load-btn:disabled {
  background: #9ca3af;
  cursor: not-allowed;
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

/* Field Info Styles */
.field-info {
  margin-top: 1rem;
  padding: 0.75rem;
  background: #f0f9ff;
  border-left: 3px solid #3b82f6;
  border-radius: 4px;
}

.field-info .info-text {
  font-size: 0.875rem;
  color: #1e40af;
  margin: 0;
  line-height: 1.5;
}

.field-info strong {
  font-weight: 600;
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

/* AI Toggle Switch Styles */
.ai-toggle-container {
  padding: 1.5rem;
  background: #f8fafc;
  border-radius: 12px;
  border: 2px solid #e2e8f0;
}

.toggle-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}

.toggle-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 600;
  color: #334155;
  font-size: 1rem;
  cursor: pointer;
}

.toggle-icon {
  font-size: 1.5rem;
}

.toggle-text {
  user-select: none;
}

.toggle-description {
  font-size: 0.875rem;
  color: #64748b;
  line-height: 1.5;
  margin: 0;
}

/* Toggle Switch */
.switch {
  position: relative;
  display: inline-block;
  width: 52px;
  height: 28px;
}

.switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #cbd5e1;
  transition: 0.4s;
  border-radius: 28px;
}

.slider:before {
  position: absolute;
  content: "";
  height: 20px;
  width: 20px;
  left: 4px;
  bottom: 4px;
  background-color: white;
  transition: 0.4s;
  border-radius: 50%;
}

input:checked + .slider {
  background-color: #3b82f6;
}

input:focus + .slider {
  box-shadow: 0 0 1px #3b82f6;
}

input:checked + .slider:before {
  transform: translateX(24px);
}

.slider:hover {
  background-color: #94a3b8;
}

input:checked + .slider:hover {
  background-color: #2563eb;
}
</style>
