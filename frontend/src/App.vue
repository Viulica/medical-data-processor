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
            @click="handleStandardProcessingTab"
            :class="{ active: activeTab === 'process' }"
            class="tab-btn"
          >
            📊 Standard
          </button>
          <button
            @click="activeTab = 'process-fast'"
            :class="{ active: activeTab === 'process-fast' }"
            class="tab-btn"
          >
            ⚡ Fast Process
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
            🏥 CPT Codes
          </button>
          <button
            @click="activeTab = 'icd'"
            :class="{ active: activeTab === 'icd' }"
            class="tab-btn"
          >
            📋 ICD Codes
          </button>
          <button
            @click="activeTab = 'modifiers'"
            :class="{ active: activeTab === 'modifiers' }"
            class="tab-btn"
          >
            💊 Modifiers + Blocks
          </button>
          <button
            @click="activeTab = 'insurance'"
            :class="{ active: activeTab === 'insurance' }"
            class="tab-btn"
          >
            🏥 Sorting
          </button>
          <div class="dropdown-container">
            <button
              @click="toggleConvertersDropdown"
              :class="{
                active: ['uni', 'instructions'].includes(activeTab),
              }"
              class="tab-btn dropdown-btn"
            >
              🔄 Converters
              <span class="dropdown-arrow">{{
                showConvertersDropdown ? "▲" : "▼"
              }}</span>
            </button>
            <div v-if="showConvertersDropdown" class="dropdown-menu">
              <button @click="selectTab('uni')" class="dropdown-item">
                🔄 Convert UNI CSV
              </button>
              <button @click="selectTab('instructions')" class="dropdown-item">
                📋 Convert Instructions
              </button>
            </div>
          </div>
          <div class="dropdown-container">
            <button
              @click="toggleSettingsDropdown"
              :class="{
                active: [
                  'config',
                  'insurance-config',
                  'templates',
                  'prediction-instructions',
                  'special-cases-templates',
                ].includes(activeTab),
              }"
              class="tab-btn dropdown-btn"
            >
              ⚙️ Settings
              <span class="dropdown-arrow">{{
                showSettingsDropdown ? "▲" : "▼"
              }}</span>
            </button>
            <div v-if="showSettingsDropdown" class="dropdown-menu">
              <button
                @click="selectTabAndLoad('config', 'loadModifiers')"
                class="dropdown-item"
              >
                ⚙️ Modifiers Config
              </button>
              <button
                @click="
                  selectTabAndLoad('insurance-config', 'loadInsuranceMappings')
                "
                class="dropdown-item"
              >
                🏥 Insurance Config
              </button>
              <button
                @click="selectTabAndLoad('templates', 'loadTemplates')"
                class="dropdown-item"
              >
                📝 Field Templates
              </button>
              <button
                @click="
                  selectTabAndLoad(
                    'prediction-instructions',
                    'loadPredictionInstructions'
                  )
                "
                class="dropdown-item"
              >
                💬 Instruction Templates
              </button>
              <button
                @click="
                  selectTabAndLoad(
                    'special-cases-templates',
                    'loadSpecialCasesTemplates'
                  )
                "
                class="dropdown-item"
              >
                🎯 Special Cases
              </button>
            </div>
          </div>
        </div>

        <!-- Process Documents Tab -->
        <div v-if="activeTab === 'process'" class="upload-section">
          <!-- Password Unlock Interface -->
          <div v-if="!isStandardProcessingUnlocked" class="section-header">
            <h2>Document Processing (Standard) - Locked 🔒</h2>
            <p>
              This processing option uses Gemini 2.5 Pro with extended thinking
              for higher accuracy. Please enter the password to unlock.
            </p>
            <div class="password-unlock-section">
              <div class="password-card">
                <div class="password-icon">🔐</div>
                <h3>Unlock Standard Processing</h3>
                <p>
                  Enter the password to access Gemini 2.5 Pro processing with
                  thinking enabled
                </p>
                <div class="password-input-group">
                  <input
                    v-model="passwordInput"
                    type="password"
                    class="password-input"
                    placeholder="Enter password"
                    @keyup.enter="unlockStandardProcessing"
                  />
                  <button @click="unlockStandardProcessing" class="unlock-btn">
                    🔓 Unlock
                  </button>
                </div>
                <div class="password-features">
                  <div class="feature-item">
                    <span class="feature-icon">🧠</span>
                    <span>Gemini 2.5 Pro with thinking enabled</span>
                  </div>
                  <div class="feature-item">
                    <span class="feature-icon">🎯</span>
                    <span>Higher accuracy for complex documents</span>
                  </div>
                  <div class="feature-item">
                    <span class="feature-icon">⏱️</span>
                    <span>Slower but more thorough processing</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Standard Processing Upload Interface (Unlocked) -->
          <div v-else>
            <div class="section-header">
              <h2>Document Processing (Standard) - Unlocked 🔓</h2>
              <p>
                Upload patient documents and processing instructions to extract
                structured medical data using Gemini 2.5 Pro with thinking
                enabled
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
                    active: isZipDragActive,
                    'has-file': zipFile,
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
                  <div v-if="!zipFile" class="dropzone-content">
                    <div class="upload-icon">📦</div>
                    <p class="upload-text">
                      Drop ZIP file here or click to browse
                    </p>
                    <p class="upload-hint">Contains patient PDF documents</p>
                  </div>
                  <div v-else class="file-preview">
                    <div class="file-icon">📦</div>
                    <div class="file-info">
                      <p class="file-name">{{ zipFile.name }}</p>
                      <p class="file-size">
                        {{ formatFileSize(zipFile.size) }}
                      </p>
                    </div>
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
                    active: isExcelDragActive,
                    'has-file': excelFile,
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
                  <div v-if="!excelFile" class="dropzone-content">
                    <div class="upload-icon">📊</div>
                    <p class="upload-text">
                      Drop Excel file here or click to browse
                    </p>
                    <p class="upload-hint">Contains field definitions</p>
                  </div>
                  <div v-else class="file-preview">
                    <div class="file-icon">📊</div>
                    <div class="file-info">
                      <p class="file-name">{{ excelFile.name }}</p>
                      <p class="file-size">
                        {{ formatFileSize(excelFile.size) }}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Step 3: Page Count -->
              <div class="upload-card">
                <div class="card-header">
                  <div class="step-number">3</div>
                  <h3>Pages Per Patient</h3>
                </div>
                <div class="page-count-selector">
                  <label for="page-count">Number of pages to process:</label>
                  <input
                    id="page-count"
                    v-model.number="pageCount"
                    type="number"
                    min="1"
                    max="50"
                    class="page-input"
                  />
                  <p class="page-hint">
                    Extract first N pages from each patient PDF (1-50)
                  </p>
                </div>
              </div>

              <!-- Step 4: OpenRouter Option -->
              <div class="upload-card">
                <div class="card-header">
                  <div class="step-number">4</div>
                  <h3>API Provider</h3>
                </div>
                <div class="template-selection-toggle">
                  <label class="toggle-label">
                    <input
                      v-model="useOpenRouterStandard"
                      type="checkbox"
                      class="toggle-checkbox"
                    />
                    <span class="toggle-text"
                      >Use OpenRouter (with web search enabled)</span
                    >
                  </label>
                  <p class="template-hint" style="margin-top: 10px">
                    When enabled, uses OpenRouter API with model:
                    <code>google/gemini-3-pro-preview:online</code>
                  </p>
                </div>
              </div>

              <!-- Step 5: Worktracker Fields (Optional) -->
              <div class="upload-card">
                <div class="card-header">
                  <div class="step-number">5</div>
                  <h3>Worktracker Info (Optional)</h3>
                </div>
                <div class="page-count-selector">
                  <div style="margin-bottom: 15px">
                    <label for="worktracker-group">Worktracker Group:</label>
                    <input
                      id="worktracker-group"
                      v-model="worktrackerGroup"
                      type="text"
                      class="page-input"
                      placeholder="Enter group name (optional)"
                    />
                  </div>
                  <div>
                    <label for="worktracker-batch">Worktracker Batch #:</label>
                    <input
                      id="worktracker-batch"
                      v-model="worktrackerBatch"
                      type="text"
                      class="page-input"
                      placeholder="Enter batch number (optional)"
                    />
                  </div>
                  <p class="page-hint" style="margin-top: 10px">
                    These values will be added as columns in the output Excel
                  </p>
                </div>
              </div>
            </div>

            <div class="action-section">
              <button
                @click="startProcessing"
                :disabled="!canProcess || isProcessing"
                class="process-btn"
                :class="{ processing: isProcessing }"
              >
                <span v-if="!isProcessing" class="btn-icon">🚀</span>
                <span v-else class="btn-icon spinning">⏳</span>
                {{
                  isProcessing ? "Processing..." : "Start Standard Processing"
                }}
              </button>
              <button v-if="!isProcessing" @click="resetForm" class="reset-btn">
                <span class="btn-icon">🔄</span>
                Reset Form
              </button>
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
              <div class="download-format-group">
                <button @click="downloadResults('csv')" class="download-btn">
                  <span class="btn-icon">📥</span>
                  Download CSV
                </button>
                <button
                  @click="downloadResults('xlsx')"
                  class="download-btn download-btn-alt"
                >
                  <span class="btn-icon">📊</span>
                  Download XLSX
                </button>
              </div>
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

            <!-- Step 2: Excel File Upload or Template Selection -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">2</div>
                <h3>Processing Template</h3>
              </div>

              <!-- Template Selection Option -->
              <div class="template-selection-toggle">
                <label class="toggle-label">
                  <input
                    v-model="useTemplateInsteadOfExcelFast"
                    type="checkbox"
                    class="toggle-checkbox"
                  />
                  <span class="toggle-text">Use saved template instead</span>
                </label>
              </div>

              <!-- Template Dropdown (when using template) -->
              <div
                v-if="useTemplateInsteadOfExcelFast"
                class="template-dropdown-section"
              >
                <select
                  v-model="selectedTemplateIdFast"
                  class="template-select"
                  @focus="ensureTemplatesLoaded"
                >
                  <option :value="null" disabled>Select a template...</option>
                  <option
                    v-for="template in allTemplatesForDropdown"
                    :key="template.id"
                    :value="template.id"
                  >
                    {{ template.name }}
                  </option>
                </select>
                <p class="template-hint">
                  <a
                    href="#"
                    @click.prevent="loadTemplates"
                    class="manage-link"
                  >
                    Manage templates
                  </a>
                </p>
              </div>

              <!-- Excel File Upload (when not using template) -->
              <div
                v-else
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

            <!-- Step 4: Worktracker Fields (Optional) -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">4</div>
                <h3>Worktracker Info (Optional)</h3>
              </div>
              <div class="settings-content">
                <div class="setting-group">
                  <label for="worktracker-group-fast">Worktracker Group</label>
                  <div class="input-wrapper">
                    <input
                      id="worktracker-group-fast"
                      v-model="worktrackerGroupFast"
                      type="text"
                      class="page-input"
                      placeholder="Enter group name (optional)"
                    />
                  </div>
                  <small class="help-text">
                    Optional: Will be added as a column in output
                  </small>
                </div>
                <div class="setting-group" style="margin-top: 15px">
                  <label for="worktracker-batch-fast"
                    >Worktracker Batch #</label
                  >
                  <div class="input-wrapper">
                    <input
                      id="worktracker-batch-fast"
                      v-model="worktrackerBatchFast"
                      type="text"
                      class="page-input"
                      placeholder="Enter batch number (optional)"
                    />
                  </div>
                  <small class="help-text">
                    Optional: Will be added as a column in output
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
              <div class="download-format-group">
                <button
                  @click="downloadResultsFast('csv')"
                  class="download-btn"
                >
                  <span class="btn-icon">📥</span>
                  Download CSV
                </button>
                <button
                  @click="downloadResultsFast('xlsx')"
                  class="download-btn download-btn-alt"
                >
                  <span class="btn-icon">📊</span>
                  Download XLSX
                </button>
              </div>
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
            <p v-if="!useVisionPrediction">
              Upload a CSV or XLSX file with a 'Procedure Description' column to
              predict anesthesia CPT codes using AI
            </p>
            <p v-else>
              Upload a ZIP file containing patient PDFs to predict anesthesia
              CPT codes by analyzing PDF pages with AI vision
            </p>
          </div>

          <!-- Vision Mode Toggle -->
          <div class="upload-card" style="margin-bottom: 20px">
            <div class="settings-content">
              <div class="form-group">
                <label class="checkbox-label">
                  <input
                    type="checkbox"
                    v-model="useVisionPrediction"
                    class="checkbox-input"
                  />
                  <span class="checkbox-text">
                    📸 Predict ASA codes from PDF pages (Vision Mode - uses
                    GPT-5)
                  </span>
                </label>
                <p
                  class="form-hint"
                  v-if="useVisionPrediction"
                  style="margin-top: 10px; color: #6b7280"
                >
                  Vision mode analyzes actual PDF pages instead of extracted
                  text. Great for handwritten notes or complex document layouts.
                </p>
              </div>
            </div>
          </div>

          <div class="upload-grid">
            <!-- CSV File Upload (Traditional Mode) -->
            <div v-if="!useVisionPrediction" class="upload-card">
              <div class="card-header">
                <div class="step-number">1</div>
                <h3>CSV/XLSX File with Procedures</h3>
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
                  accept=".csv,.xlsx,.xls"
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
                    Drag & drop CSV or XLSX file here<br />or click to browse
                  </p>
                </div>
              </div>
            </div>

            <!-- ZIP File Upload (Vision Mode) -->
            <div v-if="useVisionPrediction" class="upload-card">
              <div class="card-header">
                <div class="step-number">1</div>
                <h3>ZIP File with Patient PDFs</h3>
              </div>
              <div
                class="dropzone"
                :class="{
                  active: isVisionZipDragActive,
                  'has-file': visionZipFile,
                }"
                @drop="onVisionZipDrop"
                @dragover.prevent
                @dragenter.prevent
                @click="triggerVisionZipUpload"
              >
                <input
                  ref="visionZipInput"
                  type="file"
                  accept=".zip"
                  @change="onVisionZipFileSelect"
                  style="display: none"
                />
                <div class="upload-content">
                  <div class="upload-icon">📦</div>
                  <div v-if="visionZipFile" class="file-info">
                    <div class="file-icon">📄</div>
                    <span class="file-name">{{ visionZipFile.name }}</span>
                    <span class="file-size">{{
                      formatFileSize(visionZipFile.size)
                    }}</span>
                  </div>
                  <p v-else class="upload-text">
                    Drag & drop ZIP file here<br />or click to browse
                  </p>
                </div>
              </div>
            </div>

            <!-- Vision Page Count -->
            <div v-if="useVisionPrediction" class="upload-card">
              <div class="card-header">
                <div class="step-number">2</div>
                <h3>Pages to Analyze</h3>
              </div>
              <div class="page-count-selector">
                <label for="vision-page-count"
                  >Number of PDF pages per patient:</label
                >
                <input
                  id="vision-page-count"
                  v-model.number="visionPageCount"
                  type="number"
                  min="1"
                  max="50"
                  class="page-input"
                />
                <p class="page-hint">
                  AI will analyze the first N pages from each patient PDF (1-50)
                </p>
              </div>
            </div>

            <!-- Custom Instructions for Vision Mode -->
            <div v-if="useVisionPrediction" class="upload-card">
              <div class="card-header">
                <div class="step-number">3</div>
                <h3>Custom Coding Instructions</h3>
              </div>
              <div class="settings-content">
                <div class="form-group">
                  <label
                    for="cpt-vision-custom-instructions"
                    class="form-label"
                  >
                    Additional instructions for CPT code prediction (optional):
                  </label>
                  <textarea
                    id="cpt-vision-custom-instructions"
                    v-model="cptVisionCustomInstructions"
                    class="form-textarea"
                    rows="4"
                    placeholder="Enter specific coding guidelines, corrections, or preferences that should be applied to the AI predictions..."
                  ></textarea>
                  <p class="form-hint" style="margin-top: 8px; color: #6b7280">
                    These instructions will be appended to the AI prompt for all
                    predictions in this batch.
                  </p>
                </div>
              </div>
            </div>

            <!-- Client Selection -->
            <div v-if="!useVisionPrediction" class="upload-card">
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
                    <option value="general">GENERAL (OpenAI Model)</option>
                  </select>
                </div>
              </div>
            </div>

            <!-- Custom Instructions -->
            <div v-if="!useVisionPrediction" class="upload-card">
              <div class="card-header">
                <div class="step-number">3</div>
                <h3>Custom Coding Instructions</h3>
              </div>
              <div class="settings-content">
                <!-- Template Selection Toggle -->
                <div class="template-selection-toggle">
                  <label class="toggle-label">
                    <input
                      v-model="useCptTemplateInsteadOfText"
                      type="checkbox"
                      class="toggle-checkbox"
                    />
                    <span class="toggle-text">Use saved template</span>
                  </label>
                </div>

                <!-- Template Dropdown (when using template) -->
                <div
                  v-if="useCptTemplateInsteadOfText"
                  class="template-dropdown-section"
                >
                  <select
                    v-model="selectedCptInstructionId"
                    class="template-select"
                    @focus="ensureCptInstructionsLoaded"
                  >
                    <option :value="null" disabled>
                      Select CPT template...
                    </option>
                    <option
                      v-for="instruction in predictionInstructions.filter(
                        (i) => i.instruction_type === 'cpt'
                      )"
                      :key="instruction.id"
                      :value="instruction.id"
                    >
                      {{ instruction.name }}
                    </option>
                  </select>
                  <p class="template-hint">
                    <a
                      href="#"
                      @click.prevent="loadPredictionInstructions"
                      class="manage-link"
                    >
                      Manage instruction templates
                    </a>
                  </p>
                </div>

                <!-- Manual Text Input (when not using template) -->
                <div v-else class="form-group">
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

            <!-- Requirements (Traditional Mode) -->
            <div v-if="!useVisionPrediction" class="upload-card">
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

            <!-- Requirements (Vision Mode) -->
            <div v-if="useVisionPrediction" class="upload-card">
              <div class="card-header">
                <div class="step-number">3</div>
                <h3>How It Works</h3>
              </div>
              <div class="settings-content">
                <div class="requirement-list">
                  <div class="requirement-item">
                    <span class="requirement-icon">📦</span>
                    <span>ZIP file should contain patient PDF documents</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">📸</span>
                    <span>GPT-5 will analyze PDF pages visually</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">🏥</span>
                    <span>Identifies procedures and diagnoses from images</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">⚡</span>
                    <span
                      >Predicts CPT codes in 'ASA Code' and 'Procedure Code'
                      columns</span
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
              v-if="csvFile || visionZipFile || cptJobId"
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
              <div class="download-format-group">
                <button @click="downloadCptResults('csv')" class="download-btn">
                  <span class="btn-icon">📥</span>
                  Download CSV
                </button>
                <button
                  @click="downloadCptResults('xlsx')"
                  class="download-btn download-btn-alt"
                >
                  <span class="btn-icon">📊</span>
                  Download XLSX
                </button>
              </div>
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

        <!-- ICD Prediction Tab -->
        <div v-if="activeTab === 'icd'" class="upload-section">
          <div class="section-header">
            <h2>ICD Code Prediction</h2>
            <p>
              Upload a ZIP file containing patient PDFs to predict ICD diagnosis
              codes by analyzing PDF pages with AI vision. The AI will identify
              up to 4 ICD codes sorted by relevance to the procedure.
            </p>
          </div>

          <div class="upload-grid">
            <!-- ZIP File Upload -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">1</div>
                <h3>ZIP File with Patient PDFs</h3>
              </div>
              <div
                class="dropzone"
                :class="{
                  active: isIcdZipDragActive,
                  'has-file': icdZipFile,
                }"
                @drop="onIcdZipDrop"
                @dragover.prevent
                @dragenter.prevent
                @click="triggerIcdZipUpload"
              >
                <input
                  ref="icdZipInput"
                  type="file"
                  accept=".zip"
                  @change="onIcdZipFileSelect"
                  style="display: none"
                />
                <div class="upload-content">
                  <div class="upload-icon">📦</div>
                  <div v-if="icdZipFile" class="file-info">
                    <div class="file-icon">📄</div>
                    <span class="file-name">{{ icdZipFile.name }}</span>
                    <span class="file-size">{{
                      formatFileSize(icdZipFile.size)
                    }}</span>
                  </div>
                  <p v-else class="upload-text">
                    Drag & drop ZIP file here<br />or click to browse
                  </p>
                </div>
              </div>
            </div>

            <!-- Pages to Analyze -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">2</div>
                <h3>Pages to Analyze</h3>
              </div>
              <div class="page-count-selector">
                <label for="icd-page-count"
                  >Number of PDF pages per patient:</label
                >
                <input
                  id="icd-page-count"
                  v-model.number="icdPageCount"
                  type="number"
                  min="1"
                  max="50"
                  class="page-input"
                />
                <p class="page-hint">
                  AI will analyze the first N pages from each patient PDF (1-50)
                </p>
              </div>
            </div>

            <!-- Custom Instructions for ICD -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">3</div>
                <h3>Custom Coding Instructions</h3>
              </div>
              <div class="settings-content">
                <!-- Template Selection Toggle -->
                <div class="template-selection-toggle">
                  <label class="toggle-label">
                    <input
                      v-model="useIcdTemplateInsteadOfText"
                      type="checkbox"
                      class="toggle-checkbox"
                    />
                    <span class="toggle-text">Use saved template</span>
                  </label>
                </div>

                <!-- Template Dropdown (when using template) -->
                <div
                  v-if="useIcdTemplateInsteadOfText"
                  class="template-dropdown-section"
                >
                  <select
                    v-model="selectedIcdInstructionId"
                    class="template-select"
                    @focus="ensureIcdInstructionsLoaded"
                  >
                    <option :value="null" disabled>
                      Select ICD template...
                    </option>
                    <option
                      v-for="instruction in predictionInstructions.filter(
                        (i) => i.instruction_type === 'icd'
                      )"
                      :key="instruction.id"
                      :value="instruction.id"
                    >
                      {{ instruction.name }}
                    </option>
                  </select>
                  <p class="template-hint">
                    <a
                      href="#"
                      @click.prevent="loadPredictionInstructions"
                      class="manage-link"
                    >
                      Manage instruction templates
                    </a>
                  </p>
                </div>

                <!-- Manual Text Input (when not using template) -->
                <div v-else class="form-group">
                  <label for="icd-custom-instructions" class="form-label">
                    Additional instructions for ICD code prediction (optional):
                  </label>
                  <textarea
                    id="icd-custom-instructions"
                    v-model="icdCustomInstructions"
                    class="form-textarea"
                    rows="4"
                    placeholder="Enter specific coding guidelines, corrections, or preferences that should be applied to the AI predictions..."
                  ></textarea>
                  <p class="form-hint" style="margin-top: 8px; color: #6b7280">
                    These instructions will be appended to the AI prompt for all
                    predictions in this batch.
                  </p>
                </div>
              </div>
            </div>

            <!-- How It Works -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">4</div>
                <h3>How It Works</h3>
              </div>
              <div class="settings-content">
                <div class="requirement-list">
                  <div class="requirement-item">
                    <span class="requirement-icon">📦</span>
                    <span>ZIP file should contain patient PDF documents</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">📸</span>
                    <span>GPT-5 will analyze PDF pages visually</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">🏥</span>
                    <span
                      >Identifies primary diagnosis and related ICD codes</span
                    >
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">📋</span>
                    <span
                      >Predicts up to 4 ICD codes (ICD1, ICD2, ICD3, ICD4)
                      sorted by relevance</span
                    >
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">🎯</span>
                    <span
                      >ICD1 is the primary diagnosis (main reason for
                      procedure)</span
                    >
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Action Buttons -->
          <div class="action-section">
            <button
              @click="startIcdPrediction"
              :disabled="!canPredictIcd || isPredictingIcd"
              class="process-btn"
            >
              <span v-if="isPredictingIcd" class="spinner"></span>
              <span v-else class="btn-icon">📋</span>
              {{
                isPredictingIcd
                  ? "Predicting ICD Codes..."
                  : "Start ICD Prediction"
              }}
            </button>

            <button
              v-if="icdZipFile || icdJobId"
              @click="resetIcdForm"
              class="reset-btn"
            >
              Reset
            </button>
          </div>
        </div>

        <!-- ICD Prediction Status -->
        <div v-if="icdJobStatus" class="status-section">
          <div class="status-card">
            <div class="status-header">
              <div class="status-indicator" :class="icdJobStatus.status">
                <span class="status-icon">{{ getIcdStatusIcon() }}</span>
              </div>
              <div class="status-info">
                <h3>{{ getIcdStatusTitle() }}</h3>
                <p class="status-message">{{ icdJobStatus.message }}</p>
              </div>
            </div>

            <div
              v-if="icdJobStatus.status === 'processing'"
              class="progress-section"
            >
              <div class="progress-bar">
                <div
                  class="progress-fill"
                  :style="{ width: `${icdJobStatus.progress}%` }"
                ></div>
              </div>
              <div class="progress-text">
                {{ icdJobStatus.progress }}% Complete
              </div>
              <button @click="checkIcdJobStatus" class="check-status-btn">
                <span class="btn-icon">🔄</span>
                Check Status
              </button>
            </div>

            <div
              v-if="icdJobStatus.status === 'completed'"
              class="success-section"
            >
              <div class="download-format-group">
                <button @click="downloadIcdResults('csv')" class="download-btn">
                  <span class="btn-icon">📥</span>
                  Download CSV
                </button>
                <button
                  @click="downloadIcdResults('xlsx')"
                  class="download-btn download-btn-alt"
                >
                  <span class="btn-icon">📊</span>
                  Download XLSX
                </button>
              </div>
            </div>

            <div
              v-if="icdJobStatus.status === 'failed' && icdJobStatus.error"
              class="error-section"
            >
              <div class="error-message">
                <span class="error-icon">⚠️</span>
                <span>{{ icdJobStatus.error }}</span>
              </div>
            </div>
          </div>
        </div>

        <!-- UNI Conversion Tab -->
        <div v-if="activeTab === 'uni'" class="upload-section">
          <div class="section-header">
            <h2>UNI CSV Conversion</h2>
            <p>
              Upload a UNI CSV or XLSX file to convert it using the automated
              conversion script
            </p>
          </div>

          <div class="upload-grid">
            <!-- CSV File Upload -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">1</div>
                <h3>UNI CSV/XLSX File</h3>
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
                  accept=".csv,.xlsx,.xls"
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
                    Drag & drop UNI CSV or XLSX file here<br />or click to
                    browse
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
              <div class="download-format-group">
                <button @click="downloadUniResults('csv')" class="download-btn">
                  <span class="btn-icon">📥</span>
                  Download CSV
                </button>
                <button
                  @click="downloadUniResults('xlsx')"
                  class="download-btn download-btn-alt"
                >
                  <span class="btn-icon">📊</span>
                  Download XLSX
                </button>
              </div>
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
              <div class="download-format-group">
                <button
                  @click="downloadInstructionsResults('csv')"
                  class="download-btn"
                >
                  <span class="btn-icon">📥</span>
                  Download CSV
                </button>
                <button
                  @click="downloadInstructionsResults('xlsx')"
                  class="download-btn download-btn-alt"
                >
                  <span class="btn-icon">📊</span>
                  Download XLSX
                </button>
              </div>
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

        <!-- Modifiers + Blocks Generation Tab -->
        <div v-if="activeTab === 'modifiers'" class="upload-section">
          <div class="section-header">
            <h2>Medical Modifiers + Blocks Generation</h2>
            <p>
              Upload a CSV or XLSX file with billing data to automatically
              generate medical modifiers based on provider information. Also
              generates blocks if they are properly formatted in the
              peripheral_blocks column.
            </p>
          </div>

          <div class="upload-grid">
            <!-- CSV File Upload -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">1</div>
                <h3>Billing CSV/XLSX File</h3>
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
                  accept=".csv,.xlsx,.xls"
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

            <!-- Settings -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">2</div>
                <h3>Modifier Settings</h3>
              </div>
              <div class="settings-content">
                <div class="ai-toggle-container">
                  <div class="toggle-header">
                    <label class="toggle-label">
                      <span class="toggle-icon">⚕️</span>
                      <span class="toggle-text">Enable Medical Direction</span>
                    </label>
                    <label class="switch">
                      <input type="checkbox" v-model="enableMedicalDirection" />
                      <span class="slider"></span>
                    </label>
                  </div>
                  <p class="toggle-description">
                    {{
                      enableMedicalDirection
                        ? "Medical Direction is ENABLED. Normal medical direction rules from the database will be applied."
                        : "Medical Direction is DISABLED. All medical direction rules will be treated as NO during processing."
                    }}
                  </p>
                </div>

                <div class="ai-toggle-container" style="margin-top: 1rem">
                  <div class="toggle-header">
                    <label class="toggle-label">
                      <span class="toggle-icon">🔄</span>
                      <span class="toggle-text"
                        >Generate QK Duplicate Lines</span
                      >
                    </label>
                    <label class="switch">
                      <input type="checkbox" v-model="enableQkDuplicate" />
                      <span class="slider"></span>
                    </label>
                  </div>
                  <p class="toggle-description">
                    {{
                      enableQkDuplicate
                        ? "QK Duplicate Lines are ENABLED. When QK modifier is applied, a duplicate line will be created with CRNA as Responsible Provider and QX modifier."
                        : "QK Duplicate Lines are DISABLED. Only one line will be generated per case."
                    }}
                  </p>
                </div>
              </div>
            </div>

            <!-- Requirements -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">3</div>
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
                    <span class="requirement-icon">🎯</span>
                    <span
                      >Blocks will be generated if properly formatted in
                      peripheral_blocks column</span
                    >
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">📥</span>
                    <span>Download CSV with modifiers and blocks applied</span>
                  </div>
                  <div class="requirement-item">
                    <span class="requirement-icon">🔀</span>
                    <span
                      >Duplicate lines can be generated for QK modifiers (when
                      enabled)</span
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
                  ? "Generating Modifiers + Blocks..."
                  : "Generate Modifiers + Blocks"
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
              <div class="download-format-group">
                <button
                  @click="downloadModifiersResults('csv')"
                  class="download-btn"
                >
                  <span class="btn-icon">📥</span>
                  Download CSV
                </button>
                <button
                  @click="downloadModifiersResults('xlsx')"
                  class="download-btn download-btn-alt"
                >
                  <span class="btn-icon">📊</span>
                  Download XLSX
                </button>
              </div>
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
                  accept=".csv,.xlsx,.xls"
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
                    Drag & drop insurance data CSV or XLSX<br />or click to
                    browse
                  </p>
                </div>
              </div>
              <div class="field-info">
                <p class="info-text">
                  ℹ️ File must contain: <strong>Primary Company Name</strong>,
                  <strong>Primary Company Address 1</strong> (and optionally
                  Secondary/Tertiary equivalents)
                </p>
              </div>
            </div>

            <!-- Special Cases (Upload or Template) -->
            <div class="upload-card">
              <div class="card-header">
                <div class="step-number">2</div>
                <h3>Special Cases (Optional)</h3>
              </div>

              <!-- Toggle between Upload and Template -->
              <div class="toggle-selector">
                <button
                  @click="insuranceSpecialCasesMode = 'upload'"
                  :class="{ active: insuranceSpecialCasesMode === 'upload' }"
                  class="toggle-btn"
                >
                  📤 Upload CSV
                </button>
                <button
                  @click="insuranceSpecialCasesMode = 'template'"
                  :class="{ active: insuranceSpecialCasesMode === 'template' }"
                  class="toggle-btn"
                >
                  📋 Use Template
                </button>
              </div>

              <!-- Upload Mode -->
              <div v-if="insuranceSpecialCasesMode === 'upload'">
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
                    accept=".csv,.xlsx,.xls"
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
                      Upload special cases CSV or XLSX<br />or use a saved
                      template
                    </p>
                  </div>
                </div>
              </div>

              <!-- Template Mode -->
              <div v-else class="template-selector">
                <select
                  v-model="selectedSpecialCasesTemplateId"
                  class="template-select"
                >
                  <option :value="null">Select a template...</option>
                  <option
                    v-for="template in availableSpecialCasesTemplates"
                    :key="template.id"
                    :value="template.id"
                  >
                    {{ template.name }} ({{ template.mappings.length }}
                    mappings)
                  </option>
                </select>
                <button
                  @click="loadAvailableSpecialCasesTemplates"
                  class="refresh-btn"
                  title="Refresh templates"
                >
                  🔄
                </button>
              </div>

              <div class="field-info">
                <p class="info-text">
                  ℹ️ Optional: Provide special case overrides. Format:
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
              <div class="download-format-group">
                <button
                  @click="downloadInsuranceResults('csv')"
                  class="download-btn"
                >
                  <span class="btn-icon">📥</span>
                  Download CSV
                </button>
                <button
                  @click="downloadInsuranceResults('xlsx')"
                  class="download-btn download-btn-alt"
                >
                  <span class="btn-icon">📊</span>
                  Download XLSX
                </button>
              </div>
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

        <!-- Modifiers Config Tab -->
        <div v-if="activeTab === 'config'" class="upload-section">
          <div class="section-header">
            <h2>⚙️ Modifiers Configuration</h2>
            <p>
              Manage modifier settings for different MedNet insurance codes.
              Configure Medicare modifiers and medical direction billing.
            </p>
          </div>

          <!-- Search and Add New -->
          <div class="modifiers-controls">
            <div class="search-box">
              <input
                v-model="modifierSearch"
                @input="onModifierSearchChange"
                type="text"
                placeholder="Search by MedNet Code..."
                class="search-input"
              />
            </div>
            <button @click="showAddModal = true" class="add-btn">
              ➕ Add New Modifier
            </button>
          </div>

          <!-- Modifiers Table -->
          <div v-if="modifiers.length > 0" class="modifiers-table-container">
            <table class="modifiers-table">
              <thead>
                <tr>
                  <th>MedNet Code</th>
                  <th>Medicare Modifiers</th>
                  <th>Bill Medical Direction</th>
                  <th>Last Updated</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="modifier in modifiers" :key="modifier.mednet_code">
                  <td>
                    <strong>{{ modifier.mednet_code }}</strong>
                  </td>
                  <td>
                    <span
                      :class="
                        modifier.medicare_modifiers ? 'badge-yes' : 'badge-no'
                      "
                    >
                      {{ modifier.medicare_modifiers ? "YES" : "NO" }}
                    </span>
                  </td>
                  <td>
                    <span
                      :class="
                        modifier.bill_medical_direction
                          ? 'badge-yes'
                          : 'badge-no'
                      "
                    >
                      {{ modifier.bill_medical_direction ? "YES" : "NO" }}
                    </span>
                  </td>
                  <td>{{ formatDate(modifier.updated_at) }}</td>
                  <td>
                    <button
                      @click="editModifier(modifier)"
                      class="action-btn edit-btn"
                    >
                      ✏️ Edit
                    </button>
                    <button
                      @click="deleteModifierConfirm(modifier.mednet_code)"
                      class="action-btn delete-btn"
                    >
                      🗑️ Delete
                    </button>
                  </td>
                </tr>
              </tbody>
            </table>

            <!-- Pagination Controls -->
            <div class="pagination-container">
              <div class="pagination-info">
                Showing {{ (currentPage - 1) * pageSize + 1 }} -
                {{ Math.min(currentPage * pageSize, totalModifiers) }} of
                {{ totalModifiers }} modifiers
              </div>

              <div class="pagination-controls">
                <button
                  @click="goToPage(1)"
                  :disabled="currentPage === 1"
                  class="pagination-btn"
                >
                  ⏮️ First
                </button>
                <button
                  @click="goToPage(currentPage - 1)"
                  :disabled="currentPage === 1"
                  class="pagination-btn"
                >
                  ◀️ Prev
                </button>

                <span class="page-info">
                  Page {{ currentPage }} of {{ totalPages }}
                </span>

                <button
                  @click="goToPage(currentPage + 1)"
                  :disabled="currentPage === totalPages"
                  class="pagination-btn"
                >
                  Next ▶️
                </button>
                <button
                  @click="goToPage(totalPages)"
                  :disabled="currentPage === totalPages"
                  class="pagination-btn"
                >
                  Last ⏭️
                </button>
              </div>

              <div class="page-size-selector">
                <label>Per page:</label>
                <select
                  v-model.number="pageSize"
                  @change="changePageSize(pageSize)"
                  class="page-size-select"
                >
                  <option :value="25">25</option>
                  <option :value="50">50</option>
                  <option :value="100">100</option>
                  <option :value="200">200</option>
                </select>
              </div>
            </div>
          </div>

          <div v-else-if="modifiersLoading" class="loading-section">
            <div class="spinner"></div>
            <p>Loading modifiers...</p>
          </div>

          <div v-else class="empty-state">
            <p>
              No modifiers configured yet. Click "Add New Modifier" to get
              started.
            </p>
          </div>
        </div>

        <!-- Insurance Config Tab -->
        <div v-if="activeTab === 'insurance-config'" class="upload-section">
          <div class="section-header">
            <h2>🏥 Insurance Mappings Configuration</h2>
            <p>
              Manage insurance code mappings for UNI CSV conversion. Define how
              input insurance codes map to output codes.
            </p>
          </div>

          <!-- Bulk Import Section -->
          <div class="bulk-import-section">
            <div class="bulk-import-card">
              <h3>📤 Bulk Import from CSV</h3>
              <p>
                Upload your mednet-mapping.csv file to populate the database
              </p>
              <div class="csv-format-info">
                <strong>📋 Required CSV Format:</strong>
                <div class="format-example">
                  <code>InputValue,OutputValue</code>
                  <br />
                  <code>560013,44306</code>
                  <br />
                  <code>541015,002</code>
                  <br />
                  <code>510023,TRAN</code>
                </div>
                <p class="format-note">
                  ℹ️ CSV must have headers: <strong>InputValue</strong> and
                  <strong>OutputValue</strong>
                </p>
              </div>
              <div class="bulk-import-controls">
                <input
                  type="file"
                  ref="bulkImportFileInput"
                  @change="handleBulkImportFileSelect"
                  accept=".csv"
                  style="display: none"
                />
                <button
                  @click="$refs.bulkImportFileInput.click()"
                  class="upload-btn"
                  :disabled="insuranceBulkImporting"
                >
                  📁 Select CSV File
                </button>
                <span v-if="insuranceBulkImportFile" class="file-name">
                  {{ insuranceBulkImportFile.name }}
                </span>
                <label class="checkbox-label" v-if="insuranceBulkImportFile">
                  <input type="checkbox" v-model="insuranceClearExisting" />
                  Clear existing mappings before import
                </label>
                <button
                  v-if="insuranceBulkImportFile"
                  @click="bulkImportInsuranceMappings"
                  class="import-btn"
                  :disabled="insuranceBulkImporting"
                >
                  <span v-if="insuranceBulkImporting">⏳ Importing...</span>
                  <span v-else
                    >🚀 Import {{ insuranceBulkImportFile.name }}</span
                  >
                </button>
              </div>
              <div v-if="insuranceBulkImportResult" class="import-result">
                <div
                  v-if="insuranceBulkImportResult.success"
                  class="success-message"
                >
                  ✅ Import completed!
                  <ul>
                    <li>✨ New: {{ insuranceBulkImportResult.imported }}</li>
                    <li>🔄 Updated: {{ insuranceBulkImportResult.updated }}</li>
                    <li>⏭️ Skipped: {{ insuranceBulkImportResult.skipped }}</li>
                    <li>📊 Total: {{ insuranceBulkImportResult.total }}</li>
                  </ul>
                </div>
                <div v-else class="error-message">
                  ❌ Import failed: {{ insuranceBulkImportResult.error }}
                </div>
              </div>
            </div>
          </div>

          <!-- Search and Add New -->
          <div class="modifiers-controls">
            <div class="search-box">
              <input
                v-model="insuranceSearch"
                @input="onInsuranceSearchChange"
                type="text"
                placeholder="Search by Input or Output Code (exact match)..."
                class="search-input"
              />
            </div>
            <button @click="showAddInsuranceModal = true" class="add-btn">
              ➕ Add New Mapping
            </button>
          </div>

          <!-- Insurance Mappings Table -->
          <div
            v-if="insuranceMappings.length > 0"
            class="modifiers-table-container"
          >
            <table class="modifiers-table">
              <thead>
                <tr>
                  <th>Input Code</th>
                  <th>Output Code</th>
                  <th>Description</th>
                  <th>Last Updated</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="mapping in insuranceMappings" :key="mapping.id">
                  <td>
                    <strong>{{ mapping.input_code }}</strong>
                  </td>
                  <td>
                    <strong>{{ mapping.output_code }}</strong>
                  </td>
                  <td>{{ mapping.description || "-" }}</td>
                  <td>{{ formatDate(mapping.updated_at) }}</td>
                  <td>
                    <button
                      @click="editInsuranceMapping(mapping)"
                      class="action-btn edit-btn"
                    >
                      ✏️ Edit
                    </button>
                    <button
                      @click="
                        deleteInsuranceMappingConfirm(
                          mapping.id,
                          mapping.input_code
                        )
                      "
                      class="action-btn delete-btn"
                    >
                      🗑️ Delete
                    </button>
                  </td>
                </tr>
              </tbody>
            </table>

            <!-- Pagination Controls -->
            <div class="pagination-container">
              <div class="pagination-info">
                Showing
                {{ (insuranceCurrentPage - 1) * insurancePageSize + 1 }} -
                {{
                  Math.min(
                    insuranceCurrentPage * insurancePageSize,
                    totalInsuranceMappings
                  )
                }}
                of {{ totalInsuranceMappings }} mappings
              </div>

              <div class="pagination-controls">
                <button
                  @click="goToInsurancePage(1)"
                  :disabled="insuranceCurrentPage === 1"
                  class="pagination-btn"
                >
                  ⏮️ First
                </button>
                <button
                  @click="goToInsurancePage(insuranceCurrentPage - 1)"
                  :disabled="insuranceCurrentPage === 1"
                  class="pagination-btn"
                >
                  ◀️ Prev
                </button>

                <span class="page-info">
                  Page {{ insuranceCurrentPage }} of {{ insuranceTotalPages }}
                </span>

                <button
                  @click="goToInsurancePage(insuranceCurrentPage + 1)"
                  :disabled="insuranceCurrentPage === insuranceTotalPages"
                  class="pagination-btn"
                >
                  Next ▶️
                </button>
                <button
                  @click="goToInsurancePage(insuranceTotalPages)"
                  :disabled="insuranceCurrentPage === insuranceTotalPages"
                  class="pagination-btn"
                >
                  Last ⏭️
                </button>
              </div>

              <div class="page-size-selector">
                <label>Per page:</label>
                <select
                  v-model.number="insurancePageSize"
                  @change="changeInsurancePageSize(insurancePageSize)"
                  class="page-size-select"
                >
                  <option :value="25">25</option>
                  <option :value="50">50</option>
                  <option :value="100">100</option>
                  <option :value="200">200</option>
                </select>
              </div>
            </div>
          </div>

          <div v-else-if="insuranceMappingsLoading" class="loading-section">
            <div class="spinner"></div>
            <p>Loading insurance mappings...</p>
          </div>

          <div v-else class="empty-state">
            <p>
              No insurance mappings configured yet. Click "Add New Mapping" to
              get started.
            </p>
          </div>
        </div>

        <!-- Add/Edit Insurance Mapping Modal -->
        <div
          v-if="showAddInsuranceModal || showEditInsuranceModal"
          class="modal-overlay"
          @click="closeInsuranceModals"
        >
          <div class="modal-content" @click.stop>
            <div class="modal-header">
              <h3>
                {{
                  showEditInsuranceModal
                    ? "Edit Insurance Mapping"
                    : "Add New Insurance Mapping"
                }}
              </h3>
              <button @click="closeInsuranceModals" class="close-btn">✕</button>
            </div>
            <div class="modal-body">
              <div class="form-group">
                <label>Input Code *</label>
                <input
                  v-model="currentInsuranceMapping.input_code"
                  type="text"
                  :disabled="showEditInsuranceModal"
                  class="form-input"
                  placeholder="e.g., BCBS, Aetna"
                />
                <p class="form-hint">The insurance code from the input file</p>
              </div>
              <div class="form-group">
                <label>Output Code *</label>
                <input
                  v-model="currentInsuranceMapping.output_code"
                  type="text"
                  class="form-input"
                  placeholder="e.g., Blue Cross Blue Shield"
                />
                <p class="form-hint">The insurance code to use in the output</p>
              </div>
              <div class="form-group">
                <label>Description (Optional)</label>
                <textarea
                  v-model="currentInsuranceMapping.description"
                  class="form-textarea"
                  rows="3"
                  placeholder="Optional notes about this mapping..."
                ></textarea>
              </div>
            </div>
            <div class="modal-footer">
              <button @click="closeInsuranceModals" class="btn-secondary">
                Cancel
              </button>
              <button
                @click="saveInsuranceMapping"
                class="btn-primary"
                :disabled="
                  !currentInsuranceMapping.input_code ||
                  !currentInsuranceMapping.output_code
                "
              >
                {{ showEditInsuranceModal ? "Update" : "Create" }}
              </button>
            </div>
          </div>
        </div>

        <!-- Special Cases Templates Tab -->
        <div
          v-if="activeTab === 'special-cases-templates'"
          class="upload-section"
        >
          <div class="section-header">
            <h2>🎯 Special Cases Templates</h2>
            <p>
              Manage special case mappings for insurance sorting. Create
              reusable templates for company name to MedNet code overrides.
            </p>
          </div>

          <!-- Search and Upload -->
          <div class="modifiers-controls">
            <div class="search-box">
              <input
                v-model="specialCasesTemplateSearch"
                @input="onSpecialCasesTemplateSearchChange"
                type="text"
                placeholder="Search templates..."
                class="search-input"
              />
            </div>
            <button
              @click="showUploadSpecialCasesTemplateModal = true"
              class="add-btn"
            >
              ➕ Upload New Template
            </button>
          </div>

          <!-- Templates Table -->
          <div
            v-if="specialCasesTemplates.length > 0"
            class="modifiers-table-container"
          >
            <table class="modifiers-table">
              <thead>
                <tr>
                  <th>Template Name</th>
                  <th>Description</th>
                  <th>Mappings</th>
                  <th>Last Updated</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="template in specialCasesTemplates"
                  :key="template.id"
                >
                  <td>
                    <strong>{{ template.name }}</strong>
                  </td>
                  <td>{{ template.description || "-" }}</td>
                  <td>{{ template.mappings.length }} mappings</td>
                  <td>{{ formatDate(template.updated_at) }}</td>
                  <td>
                    <button
                      @click="viewSpecialCasesTemplate(template)"
                      class="action-btn view-btn"
                    >
                      👁️ View
                    </button>
                    <button
                      @click="editSpecialCasesTemplateMappings(template)"
                      class="action-btn edit-btn"
                    >
                      ✏️ Edit Mappings
                    </button>
                    <button
                      @click="
                        deleteSpecialCasesTemplateConfirm(
                          template.id,
                          template.name
                        )
                      "
                      class="action-btn delete-btn"
                    >
                      🗑️ Delete
                    </button>
                  </td>
                </tr>
              </tbody>
            </table>

            <!-- Pagination -->
            <div class="pagination-container">
              <div class="pagination-info">
                Showing
                {{
                  (specialCasesTemplateCurrentPage - 1) *
                    specialCasesTemplatePageSize +
                  1
                }}
                -
                {{
                  Math.min(
                    specialCasesTemplateCurrentPage *
                      specialCasesTemplatePageSize,
                    totalSpecialCasesTemplates
                  )
                }}
                of {{ totalSpecialCasesTemplates }} templates
              </div>

              <div class="pagination-controls">
                <button
                  @click="
                    specialCasesTemplateCurrentPage > 1 &&
                      loadSpecialCasesTemplates(false)
                  "
                  :disabled="specialCasesTemplateCurrentPage === 1"
                  class="pagination-btn"
                >
                  ← Previous
                </button>
                <span class="page-number">
                  Page {{ specialCasesTemplateCurrentPage }} of
                  {{ specialCasesTemplateTotalPages }}
                </span>
                <button
                  @click="
                    specialCasesTemplateCurrentPage <
                      specialCasesTemplateTotalPages &&
                      ((specialCasesTemplateCurrentPage += 1),
                      loadSpecialCasesTemplates(false))
                  "
                  :disabled="
                    specialCasesTemplateCurrentPage ===
                    specialCasesTemplateTotalPages
                  "
                  class="pagination-btn"
                >
                  Next →
                </button>
              </div>
            </div>
          </div>

          <!-- Empty State -->
          <div v-else class="empty-state">
            <p>
              No special cases templates found. Upload a CSV to create your
              first template.
            </p>
          </div>
        </div>

        <!-- Upload Special Cases Template Modal -->
        <div
          v-if="showUploadSpecialCasesTemplateModal"
          class="modal-overlay"
          @click="closeSpecialCasesTemplateModals"
        >
          <div class="modal-content" @click.stop>
            <div class="modal-header">
              <h3>Upload Special Cases Template</h3>
              <button
                @click="closeSpecialCasesTemplateModals"
                class="close-btn"
              >
                ✕
              </button>
            </div>
            <div class="modal-body">
              <div class="form-group">
                <label>Template Name *</label>
                <input
                  v-model="specialCasesTemplateUploadName"
                  type="text"
                  placeholder="e.g., Medicare Overrides 2024"
                  class="form-input"
                />
              </div>
              <div class="form-group">
                <label>Description</label>
                <textarea
                  v-model="specialCasesTemplateUploadDescription"
                  placeholder="Optional description..."
                  class="form-textarea"
                  rows="3"
                ></textarea>
              </div>
              <div class="form-group">
                <label>CSV File *</label>
                <div class="csv-format-info">
                  <strong>📋 Required CSV Format:</strong>
                  <div class="format-example">
                    <code>Company name,Mednet code</code>
                    <br />
                    <code>Medicare of LA 2024,301</code>
                    <br />
                    <code>Medicare of LA,301</code>
                  </div>
                </div>
                <input
                  type="file"
                  ref="specialCasesTemplateFileInput"
                  @change="handleSpecialCasesTemplateFileSelect"
                  accept=".csv"
                  class="file-input"
                />
                <span v-if="specialCasesTemplateFile" class="file-name">
                  {{ specialCasesTemplateFile.name }}
                </span>
              </div>
            </div>
            <div class="modal-footer">
              <button
                @click="closeSpecialCasesTemplateModals"
                class="btn-secondary"
              >
                Cancel
              </button>
              <button
                @click="uploadSpecialCasesTemplate"
                class="btn-primary"
                :disabled="
                  !specialCasesTemplateUploadName || !specialCasesTemplateFile
                "
              >
                📤 Upload Template
              </button>
            </div>
          </div>
        </div>

        <!-- View Special Cases Template Modal -->
        <div
          v-if="showViewSpecialCasesTemplateModal"
          class="modal-overlay"
          @click="closeSpecialCasesTemplateModals"
        >
          <div class="modal-content large-modal" @click.stop>
            <div class="modal-header">
              <h3>{{ currentSpecialCasesTemplate.name }}</h3>
              <button
                @click="closeSpecialCasesTemplateModals"
                class="close-btn"
              >
                ✕
              </button>
            </div>
            <div class="modal-body">
              <div class="template-info">
                <p>
                  <strong>Description:</strong>
                  {{
                    currentSpecialCasesTemplate.description || "No description"
                  }}
                </p>
                <p>
                  <strong>Total Mappings:</strong>
                  {{ currentSpecialCasesTemplate.mappings.length }}
                </p>
              </div>
              <div class="mappings-list">
                <table class="modifiers-table">
                  <thead>
                    <tr>
                      <th>Company Name</th>
                      <th>MedNet Code</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="(
                        mapping, index
                      ) in currentSpecialCasesTemplate.mappings"
                      :key="index"
                    >
                      <td>{{ mapping.company_name }}</td>
                      <td>
                        <strong>{{ mapping.mednet_code }}</strong>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            <div class="modal-footer">
              <button
                @click="closeSpecialCasesTemplateModals"
                class="btn-primary"
              >
                Close
              </button>
            </div>
          </div>
        </div>

        <!-- Edit Special Cases Mappings Modal -->
        <div
          v-if="showEditSpecialCasesMappingsModal"
          class="modal-overlay"
          @click="closeSpecialCasesTemplateModals"
        >
          <div class="modal-content large-modal" @click.stop>
            <div class="modal-header">
              <h3>Edit Mappings: {{ currentSpecialCasesTemplate.name }}</h3>
              <button
                @click="closeSpecialCasesTemplateModals"
                class="close-btn"
              >
                ✕
              </button>
            </div>
            <div class="modal-body">
              <div class="mappings-editor">
                <button
                  @click="addNewSpecialCasesMapping"
                  class="add-btn small"
                >
                  ➕ Add New Mapping
                </button>
                <div
                  v-for="(mapping, index) in editingSpecialCasesMappings"
                  :key="index"
                  class="mapping-row"
                >
                  <input
                    v-model="mapping.company_name"
                    type="text"
                    placeholder="Company Name"
                    class="form-input"
                  />
                  <input
                    v-model="mapping.mednet_code"
                    type="text"
                    placeholder="MedNet Code"
                    class="form-input"
                  />
                  <button
                    @click="removeSpecialCasesMapping(index)"
                    class="delete-btn small"
                  >
                    🗑️
                  </button>
                </div>
              </div>
            </div>
            <div class="modal-footer">
              <button
                @click="closeSpecialCasesTemplateModals"
                class="btn-secondary"
              >
                Cancel
              </button>
              <button @click="saveSpecialCasesMappings" class="btn-primary">
                💾 Save Changes
              </button>
            </div>
          </div>
        </div>


        <!-- Templates Manager Tab -->
        <div v-if="activeTab === 'templates'" class="upload-section">
          <div class="section-header">
            <h2>📝 Instruction Templates Manager</h2>
            <p>
              Manage instruction templates for PDF extraction. Upload Excel
              files as reusable templates or use existing templates for
              processing.
            </p>
          </div>

          <!-- Search and Add New -->
          <div class="modifiers-controls">
            <div class="search-box">
              <input
                v-model="templateSearch"
                @input="onTemplateSearchChange"
                type="text"
                placeholder="Search templates..."
                class="search-input"
              />
            </div>
            <button @click="showAddTemplateModal = true" class="add-btn">
              ➕ Upload New Template
            </button>
          </div>

          <!-- Templates Grid -->
          <div v-if="templates.length > 0" class="templates-grid">
            <div
              v-for="template in templates"
              :key="template.id"
              class="template-card"
            >
              <div class="template-card-header">
                <h3>{{ template.name }}</h3>
                <div class="template-card-actions">
                  <button
                    @click="viewTemplateDetails(template)"
                    class="action-btn edit-btn"
                    title="Edit"
                  >
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    >
                      <path
                        d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"
                      ></path>
                      <path
                        d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"
                      ></path>
                    </svg>
                  </button>
                  <button
                    @click="exportTemplate(template.id, template.name)"
                    class="action-btn download-btn"
                    title="Export to Excel"
                  >
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    >
                      <path
                        d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"
                      ></path>
                      <polyline points="7 10 12 15 17 10"></polyline>
                      <line x1="12" y1="15" x2="12" y2="3"></line>
                    </svg>
                  </button>
                  <button
                    @click="deleteTemplateConfirm(template.id, template.name)"
                    class="action-btn delete-btn"
                    title="Delete"
                  >
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="#dc2626"
                      stroke-width="2"
                      stroke-linecap="round"
                      stroke-linejoin="round"
                    >
                      <polyline points="3 6 5 6 21 6"></polyline>
                      <path
                        d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"
                      ></path>
                    </svg>
                  </button>
                </div>
              </div>
              <div class="template-card-body">
                <p class="template-description">
                  {{ template.description || "No description provided" }}
                </p>
                <div class="template-meta">
                  <span class="template-date">
                    📅 Created: {{ formatDate(template.created_at) }}
                  </span>
                  <span
                    v-if="template.updated_at !== template.created_at"
                    class="template-date"
                  >
                    🔄 Updated: {{ formatDate(template.updated_at) }}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <!-- Pagination Controls -->
          <div v-if="templates.length > 0" class="pagination-container">
            <div class="pagination-info">
              Showing {{ (templateCurrentPage - 1) * templatePageSize + 1 }} -
              {{
                Math.min(templateCurrentPage * templatePageSize, totalTemplates)
              }}
              of {{ totalTemplates }} templates
            </div>

            <div class="pagination-controls">
              <button
                @click="goToTemplatePage(1)"
                :disabled="templateCurrentPage === 1"
                class="pagination-btn"
              >
                ⏮️ First
              </button>
              <button
                @click="goToTemplatePage(templateCurrentPage - 1)"
                :disabled="templateCurrentPage === 1"
                class="pagination-btn"
              >
                ◀️ Prev
              </button>

              <span class="page-info">
                Page {{ templateCurrentPage }} of {{ templateTotalPages }}
              </span>

              <button
                @click="goToTemplatePage(templateCurrentPage + 1)"
                :disabled="templateCurrentPage === templateTotalPages"
                class="pagination-btn"
              >
                Next ▶️
              </button>
              <button
                @click="goToTemplatePage(templateTotalPages)"
                :disabled="templateCurrentPage === templateTotalPages"
                class="pagination-btn"
              >
                Last ⏭️
              </button>
            </div>

            <div class="page-size-selector">
              <label>Per page:</label>
              <select
                v-model.number="templatePageSize"
                @change="changeTemplatePageSize(templatePageSize)"
                class="page-size-select"
              >
                <option :value="12">12</option>
                <option :value="24">24</option>
                <option :value="48">48</option>
              </select>
            </div>
          </div>

          <div v-else-if="templatesLoading" class="loading-section">
            <div class="spinner"></div>
            <p>Loading templates...</p>
          </div>

          <div v-else class="empty-state">
            <p>
              No templates found. Click "Upload New Template" to get started.
            </p>
          </div>
        </div>

        <!-- Add Template Modal -->
        <div
          v-if="showAddTemplateModal"
          class="modal-overlay"
          @click="closeTemplateModals"
        >
          <div class="modal-content" @click.stop>
            <div class="modal-header">
              <h3>📤 Upload New Template</h3>
              <button @click="closeTemplateModals" class="close-btn">✕</button>
            </div>
            <div class="modal-body">
              <div class="form-group">
                <label>Template Name *</label>
                <input
                  v-model="currentTemplate.name"
                  type="text"
                  class="form-input"
                  placeholder="e.g., Hospital A - General Surgery"
                />
              </div>
              <div class="form-group">
                <label>Description</label>
                <textarea
                  v-model="currentTemplate.description"
                  class="form-textarea"
                  rows="3"
                  placeholder="Optional description of this template..."
                ></textarea>
              </div>
              <div class="form-group">
                <label>Excel File *</label>
                <input
                  ref="templateExcelInput"
                  type="file"
                  accept=".xlsx,.xls"
                  @change="onTemplateExcelSelect"
                  class="form-file-input"
                />
                <p v-if="currentTemplate.file" class="file-selected">
                  📄 {{ currentTemplate.file.name }}
                </p>
              </div>
            </div>
            <div class="modal-footer">
              <button @click="closeTemplateModals" class="btn-secondary">
                Cancel
              </button>
              <button
                @click="saveTemplate"
                class="btn-primary"
                :disabled="
                  !currentTemplate.name ||
                  !currentTemplate.file ||
                  isUploadingTemplate
                "
              >
                <span v-if="isUploadingTemplate">Uploading...</span>
                <span v-else>Upload Template</span>
              </button>
            </div>
          </div>
        </div>

        <!-- Edit Template Modal -->
        <div
          v-if="showEditTemplateModal"
          class="modal-overlay"
          @click="closeTemplateModals"
        >
          <div class="modal-content" @click.stop>
            <div class="modal-header">
              <h3>✏️ Edit Template</h3>
              <button @click="closeTemplateModals" class="close-btn">✕</button>
            </div>
            <div class="modal-body">
              <div class="form-group">
                <label>Template Name *</label>
                <input
                  v-model="currentTemplate.name"
                  type="text"
                  class="form-input"
                  placeholder="e.g., Hospital A - General Surgery"
                />
              </div>
              <div class="form-group">
                <label>Description</label>
                <textarea
                  v-model="currentTemplate.description"
                  class="form-textarea"
                  rows="3"
                  placeholder="Optional description of this template..."
                ></textarea>
              </div>
              <div class="form-group">
                <label>Replace Excel File (Optional)</label>
                <input
                  ref="templateExcelEditInput"
                  type="file"
                  accept=".xlsx,.xls"
                  @change="onTemplateExcelSelect"
                  class="form-file-input"
                />
                <p v-if="currentTemplate.file" class="file-selected">
                  📄 {{ currentTemplate.file.name }}
                </p>
                <p v-else class="file-hint">
                  Leave empty to keep existing field definitions
                </p>
              </div>
            </div>
            <div class="modal-footer">
              <button @click="closeTemplateModals" class="btn-secondary">
                Cancel
              </button>
              <button
                @click="updateTemplate"
                class="btn-primary"
                :disabled="!currentTemplate.name || isUploadingTemplate"
              >
                <span v-if="isUploadingTemplate">Updating...</span>
                <span v-else>Update Template</span>
              </button>
            </div>
          </div>
        </div>

        <!-- View/Edit Template Fields Modal -->
        <div
          v-if="showViewTemplateModal"
          class="modal-overlay"
          @click="closeTemplateModals"
        >
          <div class="modal-content modal-large" @click.stop>
            <div class="modal-header">
              <h3>Edit Template</h3>
              <button @click="closeTemplateModals" class="close-btn">✕</button>
            </div>
            <div class="modal-body">
              <div v-if="viewingTemplate">
                <div class="form-group">
                  <label>Template Name *</label>
                  <input
                    v-model="viewingTemplate.name"
                    type="text"
                    class="form-input"
                    placeholder="e.g., Hospital A - General Surgery"
                  />
                </div>
                <div class="template-detail-section">
                  <p class="template-description">
                    {{ viewingTemplate.description || "No description" }}
                  </p>
                  <div class="template-meta">
                    <span
                      >📅 Created:
                      {{ formatDate(viewingTemplate.created_at) }}</span
                    >
                    <span
                      >🔄 Updated:
                      {{ formatDate(viewingTemplate.updated_at) }}</span
                    >
                  </div>
                </div>

                <div class="template-fields-section">
                  <div class="fields-header">
                    <h4>Fields ({{ editingFields.length }})</h4>
                    <button @click="addNewField" class="add-field-btn">
                      ➕ Add Field
                    </button>
                  </div>

                  <div ref="fieldsContainer" class="fields-editor-container">
                    <div
                      v-for="(field, idx) in editingFields"
                      :key="idx"
                      class="field-editor-card"
                    >
                      <div class="field-editor-header">
                        <span class="field-number">#{{ idx + 1 }}</span>
                        <button
                          @click="deleteField(idx)"
                          class="delete-field-btn"
                          title="Delete field"
                        >
                          🗑️
                        </button>
                      </div>

                      <div class="field-editor-grid">
                        <div class="form-group">
                          <label>Field Name *</label>
                          <input
                            v-model="field.name"
                            type="text"
                            class="form-input"
                            placeholder="e.g., Patient Name"
                          />
                        </div>

                        <div class="form-group full-width">
                          <label>Output Format</label>
                          <textarea
                            v-model="field.output_format"
                            class="form-textarea"
                            rows="3"
                            placeholder="e.g., String, MM/DD/YYYY"
                          ></textarea>
                        </div>

                        <div class="form-group full-width">
                          <label>Description</label>
                          <textarea
                            v-model="field.description"
                            class="form-textarea"
                            rows="3"
                            placeholder="Description of this field"
                          ></textarea>
                        </div>

                        <div class="form-group full-width">
                          <label>Location Hint</label>
                          <textarea
                            v-model="field.location"
                            class="form-textarea"
                            rows="3"
                            placeholder="Where to find this field in the document"
                          ></textarea>
                        </div>

                        <div class="form-group priority-group">
                          <label class="checkbox-label">
                            <input
                              v-model="field.priority"
                              type="checkbox"
                              class="form-checkbox"
                            />
                            <span
                              >Priority Field (separate API call for higher
                              accuracy)</span
                            >
                          </label>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            <div class="modal-footer">
              <button @click="closeTemplateModals" class="btn-secondary">
                Cancel
              </button>
              <button
                @click="saveFieldEdits"
                class="btn-primary"
                :disabled="
                  isSavingFields ||
                  !canSaveFields ||
                  !viewingTemplate?.name ||
                  !viewingTemplate?.name.trim()
                "
              >
                <span v-if="isSavingFields">Saving...</span>
                <span v-else>Save Changes</span>
              </button>
            </div>
          </div>
        </div>

        <!-- Prediction Instructions Manager Tab -->
        <div
          v-if="activeTab === 'prediction-instructions'"
          class="upload-section"
        >
          <div class="section-header">
            <h2>💬 Prediction Instruction Templates</h2>
            <p>
              Manage instruction templates for CPT and ICD code prediction.
              Create reusable instruction sets for consistent predictions.
            </p>
          </div>

          <!-- Type Filter and Actions -->
          <div class="modifiers-controls">
            <div class="filter-group">
              <label>Filter by type:</label>
              <select
                v-model="predictionInstructionTypeFilter"
                @change="loadPredictionInstructions(false)"
                class="type-filter-select"
              >
                <option value="">All Types</option>
                <option value="cpt">CPT</option>
                <option value="icd">ICD</option>
              </select>
            </div>
            <div class="search-box">
              <input
                v-model="predictionInstructionSearch"
                @input="onPredictionInstructionSearchChange"
                type="text"
                placeholder="Search instructions..."
                class="search-input"
              />
            </div>
            <button
              @click="showAddPredictionInstructionModal = true"
              class="add-btn"
            >
              ➕ Add New Template
            </button>
          </div>

          <!-- Instructions Grid -->
          <div v-if="predictionInstructions.length > 0" class="templates-grid">
            <div
              v-for="instruction in predictionInstructions"
              :key="instruction.id"
              class="template-card"
            >
              <div class="template-card-header">
                <div class="template-title-section">
                  <h3>{{ instruction.name }}</h3>
                  <span
                    class="instruction-type-badge"
                    :class="'badge-' + instruction.instruction_type"
                  >
                    {{ instruction.instruction_type.toUpperCase() }}
                  </span>
                </div>
                <div class="template-card-actions">
                  <button
                    @click="viewPredictionInstructionDetails(instruction)"
                    class="action-btn view-btn"
                    title="View/Edit"
                  >
                    ✏️
                  </button>
                  <button
                    @click="
                      deletePredictionInstructionConfirm(
                        instruction.id,
                        instruction.name
                      )
                    "
                    class="action-btn delete-btn"
                    title="Delete"
                  >
                    🗑️
                  </button>
                </div>
              </div>
              <div class="template-card-body">
                <p class="template-description">
                  {{ instruction.description || "No description provided" }}
                </p>
                <div class="template-meta">
                  <span class="template-date">
                    📅 {{ formatDate(instruction.created_at) }}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <!-- Pagination -->
          <div
            v-if="predictionInstructions.length > 0"
            class="pagination-container"
          >
            <div class="pagination-info">
              Showing
              {{
                (predictionInstructionCurrentPage - 1) *
                  predictionInstructionPageSize +
                1
              }}
              -
              {{
                Math.min(
                  predictionInstructionCurrentPage *
                    predictionInstructionPageSize,
                  totalPredictionInstructions
                )
              }}
              of {{ totalPredictionInstructions }} templates
            </div>
            <div class="pagination-controls">
              <button
                @click="goToPredictionInstructionPage(1)"
                :disabled="predictionInstructionCurrentPage === 1"
                class="pagination-btn"
              >
                ⏮️ First
              </button>
              <button
                @click="
                  goToPredictionInstructionPage(
                    predictionInstructionCurrentPage - 1
                  )
                "
                :disabled="predictionInstructionCurrentPage === 1"
                class="pagination-btn"
              >
                ◀️ Prev
              </button>
              <span class="page-info">
                Page {{ predictionInstructionCurrentPage }} of
                {{ predictionInstructionTotalPages }}
              </span>
              <button
                @click="
                  goToPredictionInstructionPage(
                    predictionInstructionCurrentPage + 1
                  )
                "
                :disabled="
                  predictionInstructionCurrentPage ===
                  predictionInstructionTotalPages
                "
                class="pagination-btn"
              >
                Next ▶️
              </button>
              <button
                @click="
                  goToPredictionInstructionPage(predictionInstructionTotalPages)
                "
                :disabled="
                  predictionInstructionCurrentPage ===
                  predictionInstructionTotalPages
                "
                class="pagination-btn"
              >
                Last ⏭️
              </button>
            </div>
            <div class="page-size-selector">
              <label>Per page:</label>
              <select
                v-model.number="predictionInstructionPageSize"
                @change="
                  changePredictionInstructionPageSize(
                    predictionInstructionPageSize
                  )
                "
                class="page-size-select"
              >
                <option :value="12">12</option>
                <option :value="24">24</option>
                <option :value="48">48</option>
              </select>
            </div>
          </div>

          <div
            v-else-if="predictionInstructionsLoading"
            class="loading-section"
          >
            <div class="spinner"></div>
            <p>Loading instruction templates...</p>
          </div>

          <div v-else class="empty-state">
            <p>
              No instruction templates found. Click "Add New Template" to create
              one.
            </p>
          </div>
        </div>

        <!-- Add Prediction Instruction Modal -->
        <div
          v-if="showAddPredictionInstructionModal"
          class="modal-overlay"
          @click="closePredictionInstructionModals"
        >
          <div class="modal-content modal-large" @click.stop>
            <div class="modal-header">
              <h3>➕ Create Instruction Template</h3>
              <button
                @click="closePredictionInstructionModals"
                class="close-btn"
              >
                ✕
              </button>
            </div>
            <div class="modal-body">
              <div class="form-group">
                <label>Template Name *</label>
                <input
                  v-model="currentPredictionInstruction.name"
                  type="text"
                  class="form-input"
                  placeholder="e.g., Standard CPT Instructions"
                />
              </div>
              <div class="form-group">
                <label>Type *</label>
                <select
                  v-model="currentPredictionInstruction.instruction_type"
                  class="form-input"
                >
                  <option value="">Select type...</option>
                  <option value="cpt">CPT</option>
                  <option value="icd">ICD</option>
                </select>
              </div>
              <div class="form-group">
                <label>Description</label>
                <textarea
                  v-model="currentPredictionInstruction.description"
                  class="form-textarea"
                  rows="2"
                  placeholder="Optional description..."
                ></textarea>
              </div>
              <div class="form-group">
                <label>Instructions *</label>
                <textarea
                  v-model="currentPredictionInstruction.instructions_text"
                  class="form-textarea instruction-textarea"
                  rows="12"
                  placeholder="Enter prediction instructions here..."
                ></textarea>
              </div>
            </div>
            <div class="modal-footer">
              <button
                @click="closePredictionInstructionModals"
                class="btn-secondary"
              >
                Cancel
              </button>
              <button
                @click="savePredictionInstruction"
                class="btn-primary"
                :disabled="
                  !canSavePredictionInstruction || isSavingPredictionInstruction
                "
              >
                <span v-if="isSavingPredictionInstruction">Creating...</span>
                <span v-else>💾 Create Template</span>
              </button>
            </div>
          </div>
        </div>

        <!-- View/Edit Prediction Instruction Modal -->
        <div
          v-if="showViewPredictionInstructionModal"
          class="modal-overlay"
          @click="closePredictionInstructionModals"
        >
          <div class="modal-content modal-large" @click.stop>
            <div class="modal-header">
              <h3>✏️ Edit Instruction Template</h3>
              <button
                @click="closePredictionInstructionModals"
                class="close-btn"
              >
                ✕
              </button>
            </div>
            <div class="modal-body">
              <div class="form-group">
                <label>Template Name *</label>
                <input
                  v-model="currentPredictionInstruction.name"
                  type="text"
                  class="form-input"
                  placeholder="e.g., Standard CPT Instructions"
                />
              </div>
              <div class="form-group">
                <label>Type</label>
                <input
                  v-model="currentPredictionInstruction.instruction_type"
                  type="text"
                  class="form-input"
                  disabled
                />
              </div>
              <div class="form-group">
                <label>Description</label>
                <textarea
                  v-model="currentPredictionInstruction.description"
                  class="form-textarea"
                  rows="2"
                  placeholder="Optional description..."
                ></textarea>
              </div>
              <div class="form-group">
                <label>Instructions *</label>
                <textarea
                  v-model="currentPredictionInstruction.instructions_text"
                  class="form-textarea instruction-textarea"
                  rows="15"
                  placeholder="Enter prediction instructions here..."
                ></textarea>
              </div>
            </div>
            <div class="modal-footer">
              <button
                @click="closePredictionInstructionModals"
                class="btn-secondary"
              >
                Cancel
              </button>
              <button
                @click="updatePredictionInstruction"
                class="btn-primary"
                :disabled="
                  !canSavePredictionInstruction || isSavingPredictionInstruction
                "
              >
                <span v-if="isSavingPredictionInstruction">Saving...</span>
                <span v-else>💾 Save Changes</span>
              </button>
            </div>
          </div>
        </div>

        <!-- Add/Edit Modifier Modal -->
        <div
          v-if="showAddModal || showEditModal"
          class="modal-overlay"
          @click="closeModals"
        >
          <div class="modal-content" @click.stop>
            <div class="modal-header">
              <h3>
                {{ showEditModal ? "Edit Modifier" : "Add New Modifier" }}
              </h3>
              <button @click="closeModals" class="close-btn">✕</button>
            </div>
            <div class="modal-body">
              <div class="form-group">
                <label>MedNet Code</label>
                <input
                  v-model="currentModifier.mednet_code"
                  type="text"
                  :disabled="showEditModal"
                  class="form-input"
                  placeholder="e.g., 003, 0546"
                />
              </div>
              <div class="form-group">
                <label>
                  <input
                    v-model="currentModifier.medicare_modifiers"
                    type="checkbox"
                    class="form-checkbox"
                  />
                  Medicare Modifiers
                </label>
              </div>
              <div class="form-group">
                <label>
                  <input
                    v-model="currentModifier.bill_medical_direction"
                    type="checkbox"
                    class="form-checkbox"
                  />
                  Bill Medical Direction
                </label>
              </div>
            </div>
            <div class="modal-footer">
              <button @click="closeModals" class="btn-secondary">Cancel</button>
              <button
                @click="saveModifier"
                class="btn-primary"
                :disabled="!currentModifier.mednet_code"
              >
                {{ showEditModal ? "Update" : "Create" }}
              </button>
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
      worktrackerGroup: "",
      worktrackerBatch: "",
      jobId: null,
      jobStatus: null,
      isProcessing: false,
      isZipDragActive: false,
      isExcelDragActive: false,
      // Standard processing password protection
      isStandardProcessingUnlocked: false,
      passwordInput: "",
      useOpenRouterStandard: false, // Use OpenRouter for standard mode
      // Fast processing functionality
      zipFileFast: null,
      excelFileFast: null,
      pageCountFast: 2,
      worktrackerGroupFast: "",
      worktrackerBatchFast: "",
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
      // Vision-based CPT prediction from PDFs
      useVisionPrediction: false,
      visionZipFile: null,
      visionPageCount: 1,
      isVisionZipDragActive: false,
      cptVisionCustomInstructions: "",
      // ICD prediction functionality
      icdZipFile: null,
      icdPageCount: 1,
      icdJobId: null,
      icdJobStatus: null,
      isPredictingIcd: false,
      isIcdZipDragActive: false,
      icdCustomInstructions: "",
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
      enableMedicalDirection: true, // Toggle for medical direction (true = enabled, false = disabled)
      enableQkDuplicate: false, // Toggle for QK duplicate line generation (false = disabled by default)
      // Insurance sorting functionality
      insuranceDataCsv: null,
      specialCasesCsv: null,
      insuranceSpecialCasesMode: "upload", // 'upload' or 'template'
      selectedSpecialCasesTemplateId: null,
      availableSpecialCasesTemplates: [],
      insuranceJobId: null,
      insuranceJobStatus: null,
      isPredictingInsurance: false,
      isInsuranceDataDragActive: false,
      isSpecialCasesDragActive: false,
      enableAi: true, // Toggle for AI-powered insurance matching
      // Modifiers Config functionality
      modifiers: [],
      modifiersLoading: false,
      modifierSearch: "",
      modifierSearchTimeout: null,
      currentPage: 1,
      pageSize: 50,
      totalModifiers: 0,
      totalPages: 0,
      showAddModal: false,
      showEditModal: false,
      currentModifier: {
        mednet_code: "",
        medicare_modifiers: false,
        bill_medical_direction: false,
      },
      // Insurance Mappings Config functionality
      insuranceMappings: [],
      insuranceMappingsLoading: false,
      insuranceSearch: "",
      insuranceSearchTimeout: null,
      insuranceCurrentPage: 1,
      insurancePageSize: 50,
      totalInsuranceMappings: 0,
      insuranceTotalPages: 0,
      showAddInsuranceModal: false,
      showEditInsuranceModal: false,
      currentInsuranceMapping: {
        id: null,
        input_code: "",
        output_code: "",
        description: "",
      },
      insuranceBulkImportFile: null,
      insuranceBulkImporting: false,
      insuranceClearExisting: false,
      insuranceBulkImportResult: null,
      // Special Cases Templates functionality
      specialCasesTemplates: [],
      specialCasesTemplatesLoading: false,
      specialCasesTemplateSearch: "",
      specialCasesTemplateSearchTimeout: null,
      specialCasesTemplateCurrentPage: 1,
      specialCasesTemplatePageSize: 50,
      totalSpecialCasesTemplates: 0,
      specialCasesTemplateTotalPages: 0,
      showUploadSpecialCasesTemplateModal: false,
      showEditSpecialCasesTemplateModal: false,
      showViewSpecialCasesTemplateModal: false,
      currentSpecialCasesTemplate: {
        id: null,
        name: "",
        description: "",
        mappings: [],
      },
      specialCasesTemplateFile: null,
      specialCasesTemplateUploadName: "",
      specialCasesTemplateUploadDescription: "",
      showEditSpecialCasesMappingsModal: false,
      editingSpecialCasesMappings: [],
      // Templates Manager functionality
      templates: [], // Paginated templates for templates tab display
      allTemplatesForDropdown: [], // All templates for dropdown selection (not paginated)
      templatesLoading: false,
      loadingTemplatesForDropdown: false, // Flag to prevent multiple simultaneous loads
      templateSearch: "",
      templateSearchTimeout: null,
      templateCurrentPage: 1,
      templatePageSize: 12,
      totalTemplates: 0,
      templateTotalPages: 0,
      showAddTemplateModal: false,
      showEditTemplateModal: false,
      showViewTemplateModal: false,
      isUploadingTemplate: false,
      currentTemplate: {
        id: null,
        name: "",
        description: "",
        file: null,
      },
      viewingTemplate: null,
      editingFields: [],
      isSavingFields: false,
      // Template selection for processing tabs
      selectedTemplateIdFast: null,
      selectedTemplateIdStandard: null,
      useTemplateInsteadOfExcelFast: false,
      useTemplateInsteadOfExcelStandard: false,
      // Prediction Instructions Manager functionality
      predictionInstructions: [],
      predictionInstructionsLoading: false,
      predictionInstructionSearch: "",
      predictionInstructionSearchTimeout: null,
      predictionInstructionTypeFilter: "",
      predictionInstructionCurrentPage: 1,
      predictionInstructionPageSize: 12,
      totalPredictionInstructions: 0,
      predictionInstructionTotalPages: 0,
      showAddPredictionInstructionModal: false,
      showViewPredictionInstructionModal: false,
      isSavingPredictionInstruction: false,
      currentPredictionInstruction: {
        id: null,
        name: "",
        description: "",
        instruction_type: "",
        instructions_text: "",
      },
      // Template selection for prediction tabs
      selectedCptInstructionId: null,
      selectedIcdInstructionId: null,
      useCptTemplateInsteadOfText: false,
      useIcdTemplateInsteadOfText: false,
      // Dropdown states for navigation
      showConvertersDropdown: false,
      showSettingsDropdown: false,
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
      const hasTemplate = this.useTemplateInsteadOfExcelFast
        ? this.selectedTemplateIdFast !== null
        : this.excelFileFast;

      return (
        this.zipFileFast &&
        hasTemplate &&
        this.pageCountFast >= 1 &&
        this.pageCountFast <= 50
      );
    },
    canSplit() {
      return this.pdfFile && this.filterString.trim();
    },
    canPredictCpt() {
      if (this.useVisionPrediction) {
        return (
          this.visionZipFile &&
          this.visionPageCount >= 1 &&
          this.visionPageCount <= 50
        );
      } else {
        return this.csvFile;
      }
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
    canPredictIcd() {
      return (
        this.icdZipFile && this.icdPageCount >= 1 && this.icdPageCount <= 50
      );
    },
    canSaveFields() {
      // All fields must have a name
      return this.editingFields.every(
        (field) => field.name && field.name.trim()
      );
    },
    canSavePredictionInstruction() {
      return (
        this.currentPredictionInstruction.name &&
        this.currentPredictionInstruction.instruction_type &&
        this.currentPredictionInstruction.instructions_text &&
        this.currentPredictionInstruction.instructions_text.trim()
      );
    },
    canProcessCombined() {
      // At least one stage must be enabled
      if (
        !this.combinedEnableExtraction &&
        !this.combinedEnableCpt &&
        !this.combinedEnableIcd &&
        !this.combinedEnableModifiers
      ) {
        return false;
      }

      // If extraction is enabled, we need ZIP and template/excel
      if (this.combinedEnableExtraction) {
        const hasTemplate = this.combinedUseTemplate
          ? this.combinedSelectedTemplateId !== null
          : this.combinedExcelFile;

        return (
          this.combinedZipFile &&
          hasTemplate &&
          this.combinedPageCount >= 1 &&
          this.combinedPageCount <= 50
        );
      }

      // If extraction is disabled, we can't process CPT/ICD from scratch
      // (they need either extraction output or existing files)
      // For now, we require extraction to be enabled
      return false;
    },
  },
  mounted() {
    // Close dropdowns when clicking outside
    document.addEventListener("click", (e) => {
      const target = e.target;
      const isDropdownBtn = target.closest(".dropdown-btn");
      const isDropdownMenu = target.closest(".dropdown-menu");

      if (!isDropdownBtn && !isDropdownMenu) {
        this.showConvertersDropdown = false;
        this.showSettingsDropdown = false;
      }
    });
  },
  methods: {
    isValidCsvOrXlsxFile(filename) {
      if (!filename) return false;
      const lower = filename.toLowerCase();
      return (
        lower.endsWith(".csv") ||
        lower.endsWith(".xlsx") ||
        lower.endsWith(".xls")
      );
    },
    formatFileSize(bytes) {
      if (bytes === 0) return "0 Bytes";
      const k = 1024;
      const sizes = ["Bytes", "KB", "MB", "GB"];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
    },

    // Dropdown navigation methods
    toggleConvertersDropdown() {
      this.showConvertersDropdown = !this.showConvertersDropdown;
      this.showSettingsDropdown = false;
    },

    toggleSettingsDropdown() {
      this.showSettingsDropdown = !this.showSettingsDropdown;
      this.showConvertersDropdown = false;
    },

    selectTab(tabName) {
      this.activeTab = tabName;
      this.showConvertersDropdown = false;
      this.showSettingsDropdown = false;
    },

    selectTabAndLoad(tabName, methodName) {
      this.activeTab = tabName;
      this.showConvertersDropdown = false;
      this.showSettingsDropdown = false;
      if (methodName === "loadModifiers") {
        this.loadModifiers(true);
      } else if (methodName === "loadInsuranceMappings") {
        this.loadInsuranceMappings(true);
      } else if (methodName === "loadTemplates") {
        this.loadTemplates(true);
      } else if (methodName === "loadPredictionInstructions") {
        this.loadPredictionInstructions(true);
      }
    },

    async ensureTemplatesLoaded() {
      // Load ALL templates for dropdown (not paginated)
      // Always reload to ensure we have all templates regardless of current page
      if (this.loadingTemplatesForDropdown) {
        return; // Already loading, skip
      }

      this.loadingTemplatesForDropdown = true;
      try {
        // Start with a reasonable page size
        const pageSize = 100;
        let allTemplates = [];
        let currentPage = 1;
        let totalPages = 1;

        // Load all pages
        do {
          const response = await axios.get(
            joinUrl(API_BASE_URL, "api/templates"),
            {
              params: {
                page: currentPage,
                page_size: pageSize,
              },
            }
          );

          allTemplates.push(...response.data.templates);
          totalPages = response.data.total_pages;
          currentPage++;
        } while (currentPage <= totalPages);

        this.allTemplatesForDropdown = allTemplates;
      } catch (error) {
        console.error("Failed to load templates for dropdown:", error);
      } finally {
        this.loadingTemplatesForDropdown = false;
      }
    },

    async ensureCptInstructionsLoaded() {
      // Load CPT instruction templates if not already loaded
      if (
        this.predictionInstructions.filter((i) => i.instruction_type === "cpt")
          .length === 0 &&
        !this.predictionInstructionsLoading
      ) {
        try {
          const response = await axios.get(
            joinUrl(API_BASE_URL, "api/prediction-instructions"),
            {
              params: {
                instruction_type: "cpt",
                page: 1,
                page_size: 100,
              },
            }
          );
          this.predictionInstructions = response.data.instructions;
        } catch (error) {
          console.error("Failed to load CPT instructions:", error);
        }
      }
    },

    async ensureIcdInstructionsLoaded() {
      // Load ICD instruction templates if not already loaded
      if (
        this.predictionInstructions.filter((i) => i.instruction_type === "icd")
          .length === 0 &&
        !this.predictionInstructionsLoading
      ) {
        try {
          const response = await axios.get(
            joinUrl(API_BASE_URL, "api/prediction-instructions"),
            {
              params: {
                instruction_type: "icd",
                page: 1,
                page_size: 100,
              },
            }
          );
          // Merge with existing (in case CPT was already loaded)
          const existingIds = new Set(
            this.predictionInstructions.map((i) => i.id)
          );
          const newInstructions = response.data.instructions.filter(
            (i) => !existingIds.has(i.id)
          );
          this.predictionInstructions = [
            ...this.predictionInstructions,
            ...newInstructions,
          ];
        } catch (error) {
          console.error("Failed to load ICD instructions:", error);
        }
      }
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
    handleStandardProcessingTab() {
      if (!this.isStandardProcessingUnlocked) {
        // Just navigate to the tab to show password unlock interface
        this.activeTab = "process";
      } else {
        this.activeTab = "process";
      }
    },

    unlockStandardProcessing() {
      const correctPassword = "AMP_AI_2025";
      if (this.passwordInput === correctPassword) {
        this.isStandardProcessingUnlocked = true;
        this.passwordInput = "";
        this.toast.success(
          "Standard processing unlocked! You can now use Gemini 2.5 Pro with thinking."
        );
      } else {
        this.toast.error("Incorrect password. Please try again.");
        this.passwordInput = "";
      }
    },

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

      // Use OpenRouter model format if enabled, otherwise use Google GenAI
      const model = this.useOpenRouterStandard
        ? "google/gemini-3-pro-preview:online"
        : "gemini-3-pro-preview";
      formData.append("model", model);

      // Add worktracker fields if provided
      if (this.worktrackerGroup) {
        formData.append("worktracker_group", this.worktrackerGroup);
      }
      if (this.worktrackerBatch) {
        formData.append("worktracker_batch", this.worktrackerBatch);
      }

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
        const providerText = this.useOpenRouterStandard
          ? "OpenRouter"
          : "Gemini 3 Pro";
        this.toast.success(
          `Standard processing started with ${providerText}! Check the status section below.`
        );

        // Set initial status
        this.jobStatus = {
          status: "processing",
          progress: 0,
          message: `Processing started with ${providerText}...`,
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

    async downloadResults(format = "csv") {
      if (!this.jobId) return;

      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, `download/${this.jobId}?format=${format}`),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        const ext = format === "xlsx" ? "xlsx" : "csv";
        link.setAttribute("download", `patient_data_${this.jobId}.${ext}`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        this.toast.success(`${format.toUpperCase()} download started!`);
      } catch (error) {
        console.error("Download error:", error);
        this.toast.error("Failed to download results");
      }
    },

    resetForm() {
      this.zipFile = null;
      this.excelFile = null;
      this.pageCount = 2;
      this.worktrackerGroup = "";
      this.worktrackerBatch = "";
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
        this.toast.error(
          "Please upload files/select template and set a valid page count"
        );
        return;
      }

      this.isProcessingFast = true;
      this.jobStatusFast = null;

      const formData = new FormData();
      formData.append("zip_file", this.zipFileFast);

      // Add either excel file or template ID
      if (this.useTemplateInsteadOfExcelFast && this.selectedTemplateIdFast) {
        formData.append("template_id", this.selectedTemplateIdFast);
      } else {
        formData.append("excel_file", this.excelFileFast);
      }

      formData.append("n_pages", this.pageCountFast);
      formData.append("model", "gemini-flash-latest"); // Use the fast model

      // Add worktracker fields if provided
      if (this.worktrackerGroupFast) {
        formData.append("worktracker_group", this.worktrackerGroupFast);
      }
      if (this.worktrackerBatchFast) {
        formData.append("worktracker_batch", this.worktrackerBatchFast);
      }

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

    async downloadResultsFast(format = "csv") {
      if (!this.jobIdFast) return;

      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, `download/${this.jobIdFast}?format=${format}`),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        const ext = format === "xlsx" ? "xlsx" : "csv";
        link.setAttribute(
          "download",
          `patient_data_fast_${this.jobIdFast}.${ext}`
        );
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        this.toast.success(`${format.toUpperCase()} download started!`);
      } catch (error) {
        console.error("Download error:", error);
        this.toast.error("Failed to download results");
      }
    },

    resetFormFast() {
      this.zipFileFast = null;
      this.excelFileFast = null;
      this.pageCountFast = 2;
      this.worktrackerGroupFast = "";
      this.worktrackerBatchFast = "";
      this.jobIdFast = null;
      this.jobStatusFast = null;
      this.isProcessingFast = false;
      this.useTemplateInsteadOfExcelFast = false;
      this.selectedTemplateIdFast = null;
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
      if (files.length > 0 && this.isValidCsvOrXlsxFile(files[0].name)) {
        this.csvFile = files[0];
        this.toast.success("File uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid CSV or XLSX file");
      }
    },

    triggerCsvUpload() {
      this.$refs.csvInput.click();
    },

    onCsvFileSelect(e) {
      const file = e.target.files[0];
      if (file && this.isValidCsvOrXlsxFile(file.name)) {
        this.csvFile = file;
        this.toast.success("File uploaded successfully!");
      } else {
        this.toast.error("Please select a valid CSV or XLSX file");
      }
    },

    // Vision-based prediction handlers
    onVisionZipDrop(e) {
      e.preventDefault();
      this.isVisionZipDragActive = false;
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].name.endsWith(".zip")) {
        this.visionZipFile = files[0];
        this.toast.success("PDF ZIP file uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid ZIP file");
      }
    },

    triggerVisionZipUpload() {
      this.$refs.visionZipInput.click();
    },

    onVisionZipFileSelect(e) {
      const file = e.target.files[0];
      if (file && file.name.endsWith(".zip")) {
        this.visionZipFile = file;
        this.toast.success("PDF ZIP file uploaded successfully!");
      } else {
        this.toast.error("Please select a valid ZIP file");
      }
    },

    async startCptPrediction() {
      if (!this.canPredictCpt) {
        if (this.useVisionPrediction) {
          this.toast.error("Please upload a ZIP file containing PDFs");
        } else {
          this.toast.error("Please upload a CSV or XLSX file");
        }
        return;
      }

      this.isPredictingCpt = true;
      this.cptJobStatus = null;

      const formData = new FormData();

      // Route to appropriate endpoint based on vision prediction or CSV
      let uploadUrl;

      if (this.useVisionPrediction) {
        // Vision-based prediction from PDFs
        uploadUrl = joinUrl(API_BASE_URL, "predict-cpt-from-pdfs");
        formData.append("zip_file", this.visionZipFile);
        formData.append("n_pages", this.visionPageCount);
        formData.append("model", "openai/gpt-5");
        formData.append("max_workers", "5");

        // Use template or manual instructions (vision and non-vision share same templates)
        if (this.useCptTemplateInsteadOfText && this.selectedCptInstructionId) {
          formData.append(
            "instruction_template_id",
            this.selectedCptInstructionId
          );
        } else if (
          this.cptVisionCustomInstructions &&
          this.cptVisionCustomInstructions.trim()
        ) {
          formData.append(
            "custom_instructions",
            this.cptVisionCustomInstructions.trim()
          );
        }
        console.log(
          "🔧 CPT Vision Upload URL:",
          uploadUrl,
          "Pages:",
          this.visionPageCount
        );
      } else {
        // Traditional CSV-based prediction
        formData.append("csv_file", this.csvFile);

        // Add custom instructions or template
        if (this.useCptTemplateInsteadOfText && this.selectedCptInstructionId) {
          formData.append(
            "instruction_template_id",
            this.selectedCptInstructionId
          );
        } else if (this.customInstructions && this.customInstructions.trim()) {
          formData.append(
            "custom_instructions",
            this.customInstructions.trim()
          );
        }

        if (this.selectedClient === "tan-esc") {
          uploadUrl = joinUrl(API_BASE_URL, "predict-cpt-custom");
          formData.append("confidence_threshold", "0.5");
          console.log("🔧 CPT Upload URL (Custom Model):", uploadUrl);
        } else if (this.selectedClient === "general") {
          uploadUrl = joinUrl(API_BASE_URL, "predict-cpt-general");
          formData.append("model", "gpt5");
          formData.append("max_workers", "5");
          console.log("🔧 CPT Upload URL (General Model):", uploadUrl);
        } else {
          uploadUrl = joinUrl(API_BASE_URL, "predict-cpt");
          formData.append("client", this.selectedClient);
          console.log("🔧 CPT Upload URL:", uploadUrl);
        }
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

    async downloadCptResults(format = "csv") {
      if (!this.cptJobId) return;

      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, `download/${this.cptJobId}?format=${format}`),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        const ext = format === "xlsx" ? "xlsx" : "csv";
        link.setAttribute(
          "download",
          `cpt_predictions_${this.cptJobId}.${ext}`
        );
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        this.toast.success(`${format.toUpperCase()} download started!`);
      } catch (error) {
        console.error("Download error:", error);
        this.toast.error("Failed to download results");
      }
    },

    resetCptForm() {
      this.csvFile = null;
      this.visionZipFile = null;
      this.useVisionPrediction = false;
      this.visionPageCount = 1;
      this.cptVisionCustomInstructions = "";
      this.cptJobId = null;
      this.cptJobStatus = null;
      this.isPredictingCpt = false;
      this.isCsvDragActive = false;
      this.isVisionZipDragActive = false;
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

    // ICD prediction methods
    onIcdZipDrop(e) {
      e.preventDefault();
      this.isIcdZipDragActive = false;
      const files = e.dataTransfer.files;
      if (files.length > 0 && files[0].name.endsWith(".zip")) {
        this.icdZipFile = files[0];
        this.toast.success("PDF ZIP file uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid ZIP file");
      }
    },

    triggerIcdZipUpload() {
      this.$refs.icdZipInput.click();
    },

    onIcdZipFileSelect(e) {
      const file = e.target.files[0];
      if (file && file.name.endsWith(".zip")) {
        this.icdZipFile = file;
        this.toast.success("PDF ZIP file uploaded successfully!");
      } else {
        this.toast.error("Please select a valid ZIP file");
      }
    },

    async startIcdPrediction() {
      if (!this.canPredictIcd) {
        this.toast.error("Please upload a ZIP file containing PDFs");
        return;
      }

      this.isPredictingIcd = true;
      this.icdJobStatus = null;

      const formData = new FormData();
      formData.append("zip_file", this.icdZipFile);
      formData.append("n_pages", this.icdPageCount);
      formData.append("model", "openai/gpt-5");
      formData.append("max_workers", "5");

      // Use template or manual instructions
      if (this.useIcdTemplateInsteadOfText && this.selectedIcdInstructionId) {
        formData.append(
          "instruction_template_id",
          this.selectedIcdInstructionId
        );
      } else if (
        this.icdCustomInstructions &&
        this.icdCustomInstructions.trim()
      ) {
        formData.append(
          "custom_instructions",
          this.icdCustomInstructions.trim()
        );
      }

      const uploadUrl = joinUrl(API_BASE_URL, "predict-icd-from-pdfs");
      console.log("🔧 ICD Upload URL:", uploadUrl, "Pages:", this.icdPageCount);

      try {
        const response = await axios.post(uploadUrl, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        this.icdJobId = response.data.job_id;
        this.toast.success(
          "ICD prediction started! Check the status section below."
        );

        // Set initial status
        this.icdJobStatus = {
          status: "processing",
          progress: 0,
          message: "ICD prediction started...",
        };
      } catch (error) {
        console.error("ICD prediction error:", error);
        this.toast.error("Failed to start ICD prediction. Please try again.");
        this.isPredictingIcd = false;
      }
    },

    async checkIcdJobStatus() {
      if (!this.icdJobId) {
        this.toast.error("No job ID available");
        return;
      }

      try {
        const statusUrl = joinUrl(API_BASE_URL, `status/${this.icdJobId}`);
        const response = await axios.get(statusUrl);

        this.icdJobStatus = response.data;

        if (response.data.status === "completed") {
          this.toast.success("ICD prediction completed!");
          this.isPredictingIcd = false;
        } else if (response.data.status === "failed") {
          this.toast.error(
            `ICD prediction failed: ${response.data.error || "Unknown error"}`
          );
          this.isPredictingIcd = false;
        }
      } catch (error) {
        console.error("Status check error:", error);
        this.toast.error("Failed to check status");
      }
    },

    async downloadIcdResults(format = "csv") {
      if (!this.icdJobId) return;

      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, `download/${this.icdJobId}?format=${format}`),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        const ext = format === "xlsx" ? "xlsx" : "csv";
        link.setAttribute(
          "download",
          `icd_predictions_${this.icdJobId}.${ext}`
        );
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        this.toast.success(`${format.toUpperCase()} download started!`);
      } catch (error) {
        console.error("Download error:", error);
        this.toast.error("Failed to download results");
      }
    },

    resetIcdForm() {
      this.icdZipFile = null;
      this.icdPageCount = 1;
      this.icdCustomInstructions = "";
      this.icdJobId = null;
      this.icdJobStatus = null;
      this.isPredictingIcd = false;
      this.isIcdZipDragActive = false;
    },

    getIcdStatusTitle() {
      if (!this.icdJobStatus) return "";

      switch (this.icdJobStatus.status) {
        case "completed":
          return "ICD Prediction Complete";
        case "failed":
          return "ICD Prediction Failed";
        case "processing":
          return "Predicting ICD Codes";
        default:
          return "ICD Prediction Status";
      }
    },

    getIcdStatusIcon() {
      if (!this.icdJobStatus) return "";

      switch (this.icdJobStatus.status) {
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

    // UNI conversion methods
    onUniCsvDrop(e) {
      e.preventDefault();
      this.isUniCsvDragActive = false;
      const files = e.dataTransfer.files;
      if (files.length > 0 && this.isValidCsvOrXlsxFile(files[0].name)) {
        this.uniCsvFile = files[0];
        this.toast.success("UNI file uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid CSV or XLSX file");
      }
    },

    triggerUniCsvUpload() {
      this.$refs.uniCsvInput.click();
    },

    onUniCsvFileSelect(e) {
      const file = e.target.files[0];
      if (file && this.isValidCsvOrXlsxFile(file.name)) {
        this.uniCsvFile = file;
        this.toast.success("UNI file uploaded successfully!");
      } else {
        this.toast.error("Please select a valid CSV or XLSX file");
      }
    },

    async startUniConversion() {
      if (!this.canConvertUni) {
        this.toast.error("Please upload a CSV or XLSX file");
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

    async downloadUniResults(format = "csv") {
      if (!this.uniJobId) return;

      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, `download/${this.uniJobId}?format=${format}`),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        const ext = format === "xlsx" ? "xlsx" : "csv";
        link.setAttribute("download", `uni_converted_${this.uniJobId}.${ext}`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        this.toast.success(`${format.toUpperCase()} download started!`);
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

    async downloadInstructionsResults(format = "csv") {
      if (!this.instructionsJobId) return;

      try {
        const response = await axios.get(
          joinUrl(
            API_BASE_URL,
            `download/${this.instructionsJobId}?format=${format}`
          ),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        const ext = format === "xlsx" ? "xlsx" : "csv";
        link.setAttribute(
          "download",
          `instructions_converted_${this.instructionsJobId}.${ext}`
        );
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        this.toast.success(`${format.toUpperCase()} download started!`);
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
      if (files.length > 0 && this.isValidCsvOrXlsxFile(files[0].name)) {
        this.modifiersCsvFile = files[0];
        this.toast.success("Modifiers file uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid CSV or XLSX file");
      }
    },

    triggerModifiersCsvUpload() {
      this.$refs.modifiersCsvInput.click();
    },

    onModifiersCsvFileSelect(e) {
      const file = e.target.files[0];
      if (file && this.isValidCsvOrXlsxFile(file.name)) {
        this.modifiersCsvFile = file;
        this.toast.success("Modifiers file uploaded successfully!");
      } else {
        this.toast.error("Please select a valid CSV or XLSX file");
      }
    },

    async startModifiersGeneration() {
      if (!this.canGenerateModifiers) {
        this.toast.error("Please upload a CSV or XLSX file");
        return;
      }

      this.isGeneratingModifiers = true;
      this.modifiersJobStatus = null;

      const formData = new FormData();
      formData.append("csv_file", this.modifiersCsvFile);
      // Invert the logic: when enableMedicalDirection is false, turn_off_medical_direction is true
      formData.append(
        "turn_off_medical_direction",
        !this.enableMedicalDirection
      );
      // Add QK duplicate generation parameter
      formData.append("generate_qk_duplicate", this.enableQkDuplicate);

      const uploadUrl = joinUrl(API_BASE_URL, "generate-modifiers");
      console.log("🔧 Modifiers Upload URL:", uploadUrl);
      console.log("⚕️ Enable Medical Direction:", this.enableMedicalDirection);
      console.log("🔄 Enable QK Duplicate:", this.enableQkDuplicate);

      try {
        const response = await axios.post(uploadUrl, formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        this.modifiersJobId = response.data.job_id;
        const qkMsg = this.enableQkDuplicate ? " + QK Duplicates" : "";
        this.toast.success(
          `Modifiers generation started${
            !this.enableMedicalDirection ? " (Medical Direction OFF)" : ""
          }${qkMsg}! Check the status section below.`
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

    async downloadModifiersResults(format = "csv") {
      if (!this.modifiersJobId) return;

      try {
        const response = await axios.get(
          joinUrl(
            API_BASE_URL,
            `download/${this.modifiersJobId}?format=${format}`
          ),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        const ext = format === "xlsx" ? "xlsx" : "csv";
        link.setAttribute(
          "download",
          `modifiers_generated_${this.modifiersJobId}.${ext}`
        );
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        this.toast.success(`${format.toUpperCase()} download started!`);
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
      this.enableMedicalDirection = true;
      this.enableQkDuplicate = false;
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
      if (files.length > 0 && this.isValidCsvOrXlsxFile(files[0].name)) {
        this.insuranceDataCsv = files[0];
        this.toast.success("Insurance data file uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid CSV or XLSX file");
      }
    },

    onInsuranceDataCsvFileSelect(e) {
      const file = e.target.files[0];
      if (file && this.isValidCsvOrXlsxFile(file.name)) {
        this.insuranceDataCsv = file;
        this.toast.success("Insurance data file uploaded successfully!");
      } else {
        this.toast.error("Please select a valid CSV or XLSX file");
      }
    },

    triggerInsuranceDataCsvUpload() {
      this.$refs.insuranceDataCsvInput.click();
    },

    onSpecialCasesCsvDrop(e) {
      e.preventDefault();
      this.isSpecialCasesDragActive = false;
      const files = e.dataTransfer.files;
      if (files.length > 0 && this.isValidCsvOrXlsxFile(files[0].name)) {
        this.specialCasesCsv = files[0];
        this.toast.success("Special cases file uploaded successfully!");
      } else {
        this.toast.error("Please upload a valid CSV or XLSX file");
      }
    },

    onSpecialCasesCsvFileSelect(e) {
      const file = e.target.files[0];
      if (file && this.isValidCsvOrXlsxFile(file.name)) {
        this.specialCasesCsv = file;
        this.toast.success("Special cases file uploaded successfully!");
      } else {
        this.toast.error("Please select a valid CSV or XLSX file");
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

      // Handle special cases - either file upload or template
      if (
        this.insuranceSpecialCasesMode === "template" &&
        this.selectedSpecialCasesTemplateId
      ) {
        formData.append(
          "special_cases_template_id",
          this.selectedSpecialCasesTemplateId
        );
      } else if (
        this.insuranceSpecialCasesMode === "upload" &&
        this.specialCasesCsv
      ) {
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

    async loadAvailableSpecialCasesTemplates() {
      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, "api/special-cases-templates"),
          {
            params: { page: 1, page_size: 100 },
          }
        );
        this.availableSpecialCasesTemplates = response.data.templates;
      } catch (error) {
        console.error("Failed to load special cases templates:", error);
        this.toast.error("Failed to load templates");
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

    async downloadInsuranceResults(format = "csv") {
      if (!this.insuranceJobId) return;

      try {
        const response = await axios.get(
          joinUrl(
            API_BASE_URL,
            `download/${this.insuranceJobId}?format=${format}`
          ),
          {
            responseType: "blob",
          }
        );

        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        const ext = format === "xlsx" ? "xlsx" : "csv";
        link.setAttribute(
          "download",
          `insurance_codes_${this.insuranceJobId}.${ext}`
        );
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        this.toast.success(`${format.toUpperCase()} download started!`);
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

    // ========================================================================
    // Modifiers Config Methods
    // ========================================================================

    async loadModifiers(resetPage = true) {
      if (resetPage) {
        this.activeTab = "config";
        this.currentPage = 1;
      }

      this.modifiersLoading = true;
      try {
        const params = new URLSearchParams({
          page: this.currentPage.toString(),
          page_size: this.pageSize.toString(),
        });

        if (this.modifierSearch) {
          params.append("search", this.modifierSearch);
        }

        const response = await axios.get(
          joinUrl(API_BASE_URL, `api/modifiers?${params}`)
        );

        this.modifiers = response.data.modifiers || [];
        this.totalModifiers = response.data.total || 0;
        this.totalPages = response.data.total_pages || 0;
        this.currentPage = response.data.page || 1;

        if (resetPage) {
          this.toast.success(`Loaded ${this.totalModifiers} modifiers`);
        }
      } catch (error) {
        console.error("Failed to load modifiers:", error);
        this.toast.error("Failed to load modifiers from database");
        this.modifiers = [];
        this.totalModifiers = 0;
        this.totalPages = 0;
      } finally {
        this.modifiersLoading = false;
      }
    },

    onModifierSearchChange() {
      // Debounce search
      clearTimeout(this.modifierSearchTimeout);
      this.modifierSearchTimeout = setTimeout(() => {
        this.loadModifiers(true);
      }, 500);
    },

    goToPage(page) {
      if (page >= 1 && page <= this.totalPages) {
        this.currentPage = page;
        this.loadModifiers(false);
      }
    },

    changePageSize(newSize) {
      this.pageSize = newSize;
      this.currentPage = 1;
      this.loadModifiers(false);
    },

    editModifier(modifier) {
      this.currentModifier = {
        mednet_code: modifier.mednet_code,
        medicare_modifiers: modifier.medicare_modifiers,
        bill_medical_direction: modifier.bill_medical_direction,
      };
      this.showEditModal = true;
    },

    async saveModifier() {
      if (!this.currentModifier.mednet_code) {
        this.toast.error("MedNet Code is required");
        return;
      }

      try {
        const formData = new FormData();
        formData.append("mednet_code", this.currentModifier.mednet_code);
        formData.append(
          "medicare_modifiers",
          this.currentModifier.medicare_modifiers
        );
        formData.append(
          "bill_medical_direction",
          this.currentModifier.bill_medical_direction
        );

        if (this.showEditModal) {
          // Update existing
          await axios.put(
            joinUrl(
              API_BASE_URL,
              `api/modifiers/${this.currentModifier.mednet_code}`
            ),
            formData
          );
          this.toast.success("Modifier updated successfully!");
        } else {
          // Create new
          await axios.post(joinUrl(API_BASE_URL, "api/modifiers"), formData);
          this.toast.success("Modifier created successfully!");
        }

        this.closeModals();
        await this.loadModifiers(false); // Keep current page
      } catch (error) {
        console.error("Failed to save modifier:", error);
        this.toast.error("Failed to save modifier");
      }
    },

    async deleteModifierConfirm(mednetCode) {
      if (
        !confirm(`Are you sure you want to delete modifier "${mednetCode}"?`)
      ) {
        return;
      }

      try {
        await axios.delete(
          joinUrl(API_BASE_URL, `api/modifiers/${mednetCode}`)
        );
        this.toast.success("Modifier deleted successfully!");
        await this.loadModifiers(false); // Keep current page
      } catch (error) {
        console.error("Failed to delete modifier:", error);
        this.toast.error("Failed to delete modifier");
      }
    },

    closeModals() {
      this.showAddModal = false;
      this.showEditModal = false;
      this.currentModifier = {
        mednet_code: "",
        medicare_modifiers: false,
        bill_medical_direction: false,
      };
    },

    // ========================================================================
    // Insurance Mappings Config Methods
    // ========================================================================

    async loadInsuranceMappings(resetPage = true) {
      if (resetPage) {
        this.activeTab = "insurance-config";
        this.insuranceCurrentPage = 1;
      }

      this.insuranceMappingsLoading = true;
      try {
        const params = {
          page: this.insuranceCurrentPage,
          page_size: this.insurancePageSize,
        };

        if (this.insuranceSearch) {
          params.search = this.insuranceSearch;
        }

        const response = await axios.get(
          joinUrl(API_BASE_URL, "api/insurance-mappings"),
          { params }
        );

        this.insuranceMappings = response.data.mappings;
        this.totalInsuranceMappings = response.data.total;
        this.insuranceTotalPages = response.data.total_pages;
      } catch (error) {
        console.error("Failed to load insurance mappings:", error);
        this.toast.error("Failed to load insurance mappings");
      } finally {
        this.insuranceMappingsLoading = false;
      }
    },

    onInsuranceSearchChange() {
      if (this.insuranceSearchTimeout) {
        clearTimeout(this.insuranceSearchTimeout);
      }

      this.insuranceSearchTimeout = setTimeout(() => {
        this.insuranceCurrentPage = 1;
        this.loadInsuranceMappings(false);
      }, 500);
    },

    goToInsurancePage(page) {
      if (page < 1 || page > this.insuranceTotalPages) return;
      this.insuranceCurrentPage = page;
      this.loadInsuranceMappings(false);
    },

    changeInsurancePageSize(newSize) {
      this.insurancePageSize = newSize;
      this.insuranceCurrentPage = 1;
      this.loadInsuranceMappings(false);
    },

    editInsuranceMapping(mapping) {
      this.currentInsuranceMapping = {
        id: mapping.id,
        input_code: mapping.input_code,
        output_code: mapping.output_code,
        description: mapping.description || "",
      };
      this.showEditInsuranceModal = true;
    },

    async saveInsuranceMapping() {
      try {
        const formData = new FormData();
        formData.append("input_code", this.currentInsuranceMapping.input_code);
        formData.append(
          "output_code",
          this.currentInsuranceMapping.output_code
        );
        formData.append(
          "description",
          this.currentInsuranceMapping.description || ""
        );

        if (this.showEditInsuranceModal) {
          await axios.put(
            joinUrl(
              API_BASE_URL,
              `api/insurance-mappings/${this.currentInsuranceMapping.id}`
            ),
            formData
          );
          this.toast.success("Insurance mapping updated successfully!");
        } else {
          await axios.post(
            joinUrl(API_BASE_URL, "api/insurance-mappings"),
            formData
          );
          this.toast.success("Insurance mapping created successfully!");
        }

        this.closeInsuranceModals();
        await this.loadInsuranceMappings(false);
      } catch (error) {
        console.error("Failed to save insurance mapping:", error);
        this.toast.error("Failed to save insurance mapping");
      }
    },

    async deleteInsuranceMappingConfirm(mappingId, inputCode) {
      if (
        !confirm(`Are you sure you want to delete mapping for "${inputCode}"?`)
      ) {
        return;
      }

      try {
        await axios.delete(
          joinUrl(API_BASE_URL, `api/insurance-mappings/${mappingId}`)
        );
        this.toast.success("Insurance mapping deleted successfully!");
        await this.loadInsuranceMappings(false);
      } catch (error) {
        console.error("Failed to delete insurance mapping:", error);
        this.toast.error("Failed to delete insurance mapping");
      }
    },

    closeInsuranceModals() {
      this.showAddInsuranceModal = false;
      this.showEditInsuranceModal = false;
      this.currentInsuranceMapping = {
        id: null,
        input_code: "",
        output_code: "",
        description: "",
      };
    },

    handleBulkImportFileSelect(event) {
      const file = event.target.files[0];
      if (file) {
        this.insuranceBulkImportFile = file;
        this.insuranceBulkImportResult = null;
      }
    },

    async bulkImportInsuranceMappings() {
      if (!this.insuranceBulkImportFile) {
        this.toast.error("Please select a CSV file first");
        return;
      }

      this.insuranceBulkImporting = true;
      this.insuranceBulkImportResult = null;

      try {
        const formData = new FormData();
        formData.append("csv_file", this.insuranceBulkImportFile);
        formData.append("clear_existing", this.insuranceClearExisting);

        const response = await axios.post(
          joinUrl(API_BASE_URL, "api/insurance-mappings/bulk-import"),
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );

        this.insuranceBulkImportResult = {
          success: true,
          imported: response.data.imported,
          updated: response.data.updated,
          skipped: response.data.skipped,
          total: response.data.total,
        };

        this.toast.success(
          `✅ Imported ${response.data.imported} mappings, updated ${response.data.updated}`
        );

        // Reset file input and reload mappings
        this.insuranceBulkImportFile = null;
        this.insuranceClearExisting = false;
        if (this.$refs.bulkImportFileInput) {
          this.$refs.bulkImportFileInput.value = "";
        }

        // Reload the mappings table
        await this.loadInsuranceMappings(false);
      } catch (error) {
        console.error("Failed to bulk import insurance mappings:", error);
        this.insuranceBulkImportResult = {
          success: false,
          error:
            error.response?.data?.detail ||
            "Failed to import insurance mappings",
        };
        this.toast.error("Failed to import insurance mappings");
      } finally {
        this.insuranceBulkImporting = false;
      }
    },

    // ========================================================================
    // Special Cases Templates Methods
    // ========================================================================

    async loadSpecialCasesTemplates(resetPage = true) {
      if (resetPage) {
        this.activeTab = "special-cases-templates";
        this.specialCasesTemplateCurrentPage = 1;
      }

      this.specialCasesTemplatesLoading = true;
      try {
        const params = {
          page: this.specialCasesTemplateCurrentPage,
          page_size: this.specialCasesTemplatePageSize,
        };

        if (this.specialCasesTemplateSearch) {
          params.search = this.specialCasesTemplateSearch;
        }

        const response = await axios.get(
          joinUrl(API_BASE_URL, "api/special-cases-templates"),
          { params }
        );

        this.specialCasesTemplates = response.data.templates;
        this.totalSpecialCasesTemplates = response.data.total;
        this.specialCasesTemplateTotalPages = response.data.total_pages;
      } catch (error) {
        console.error("Failed to load special cases templates:", error);
        this.toast.error("Failed to load templates");
      } finally {
        this.specialCasesTemplatesLoading = false;
      }
    },

    onSpecialCasesTemplateSearchChange() {
      clearTimeout(this.specialCasesTemplateSearchTimeout);
      this.specialCasesTemplateSearchTimeout = setTimeout(() => {
        this.specialCasesTemplateCurrentPage = 1;
        this.loadSpecialCasesTemplates(false);
      }, 500);
    },

    handleSpecialCasesTemplateFileSelect(event) {
      const file = event.target.files[0];
      if (file) {
        this.specialCasesTemplateFile = file;
      }
    },

    async uploadSpecialCasesTemplate() {
      if (
        !this.specialCasesTemplateUploadName ||
        !this.specialCasesTemplateFile
      ) {
        this.toast.error("Please provide template name and CSV file");
        return;
      }

      try {
        const formData = new FormData();
        formData.append("csv_file", this.specialCasesTemplateFile);
        formData.append("name", this.specialCasesTemplateUploadName);
        formData.append(
          "description",
          this.specialCasesTemplateUploadDescription
        );

        const response = await axios.post(
          joinUrl(API_BASE_URL, "api/special-cases-templates/upload"),
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );

        this.toast.success(
          `Template "${response.data.name}" created with ${response.data.mappings_count} mappings`
        );
        this.closeSpecialCasesTemplateModals();
        await this.loadSpecialCasesTemplates(false);
      } catch (error) {
        console.error("Failed to upload special cases template:", error);
        this.toast.error(
          error.response?.data?.detail || "Failed to upload template"
        );
      }
    },

    viewSpecialCasesTemplate(template) {
      this.currentSpecialCasesTemplate = { ...template };
      this.showViewSpecialCasesTemplateModal = true;
    },

    editSpecialCasesTemplateMappings(template) {
      this.currentSpecialCasesTemplate = { ...template };
      this.editingSpecialCasesMappings = JSON.parse(
        JSON.stringify(template.mappings)
      );
      this.showEditSpecialCasesMappingsModal = true;
    },

    addNewSpecialCasesMapping() {
      this.editingSpecialCasesMappings.push({
        company_name: "",
        mednet_code: "",
      });
    },

    removeSpecialCasesMapping(index) {
      this.editingSpecialCasesMappings.splice(index, 1);
    },

    async saveSpecialCasesMappings() {
      try {
        // Filter out empty mappings
        const validMappings = this.editingSpecialCasesMappings.filter(
          (m) => m.company_name.trim() && m.mednet_code.trim()
        );

        await axios.put(
          joinUrl(
            API_BASE_URL,
            `api/special-cases-templates/${this.currentSpecialCasesTemplate.id}/mappings`
          ),
          validMappings
        );

        this.toast.success("Mappings updated successfully!");
        this.closeSpecialCasesTemplateModals();
        await this.loadSpecialCasesTemplates(false);
      } catch (error) {
        console.error("Failed to update mappings:", error);
        this.toast.error("Failed to update mappings");
      }
    },

    async deleteSpecialCasesTemplateConfirm(templateId, templateName) {
      if (
        !confirm(`Are you sure you want to delete template "${templateName}"?`)
      ) {
        return;
      }

      try {
        await axios.delete(
          joinUrl(API_BASE_URL, `api/special-cases-templates/${templateId}`)
        );
        this.toast.success("Template deleted successfully!");
        await this.loadSpecialCasesTemplates(false);
      } catch (error) {
        console.error("Failed to delete template:", error);
        this.toast.error("Failed to delete template");
      }
    },

    closeSpecialCasesTemplateModals() {
      this.showUploadSpecialCasesTemplateModal = false;
      this.showViewSpecialCasesTemplateModal = false;
      this.showEditSpecialCasesMappingsModal = false;
      this.specialCasesTemplateFile = null;
      this.specialCasesTemplateUploadName = "";
      this.specialCasesTemplateUploadDescription = "";
      this.currentSpecialCasesTemplate = {
        id: null,
        name: "",
        description: "",
        mappings: [],
      };
      this.editingSpecialCasesMappings = [];
      if (this.$refs.specialCasesTemplateFileInput) {
        this.$refs.specialCasesTemplateFileInput.value = "";
      }
    },



    // ========================================================================
    // Templates Manager Methods
    // ========================================================================

    async loadTemplates(resetPage = true) {
      if (resetPage) {
        this.activeTab = "templates";
        this.templateCurrentPage = 1;
      }

      this.templatesLoading = true;
      try {
        const params = {
          page: this.templateCurrentPage,
          page_size: this.templatePageSize,
        };

        if (this.templateSearch) {
          params.search = this.templateSearch;
        }

        const response = await axios.get(
          joinUrl(API_BASE_URL, "api/templates"),
          { params }
        );

        this.templates = response.data.templates;
        this.totalTemplates = response.data.total;
        this.templateTotalPages = response.data.total_pages;
      } catch (error) {
        console.error("Failed to load templates:", error);
        this.toast.error("Failed to load templates");
      } finally {
        this.templatesLoading = false;
      }
    },

    onTemplateSearchChange() {
      if (this.templateSearchTimeout) {
        clearTimeout(this.templateSearchTimeout);
      }

      this.templateSearchTimeout = setTimeout(() => {
        this.templateCurrentPage = 1;
        this.loadTemplates(false);
      }, 500);
    },

    goToTemplatePage(page) {
      if (page < 1 || page > this.templateTotalPages) return;
      this.templateCurrentPage = page;
      this.loadTemplates(false);
    },

    changeTemplatePageSize(newSize) {
      this.templatePageSize = newSize;
      this.templateCurrentPage = 1;
      this.loadTemplates(false);
    },

    onTemplateExcelSelect(event) {
      const file = event.target.files[0];
      if (file) {
        this.currentTemplate.file = file;
      }
    },

    async saveTemplate() {
      if (!this.currentTemplate.name || !this.currentTemplate.file) {
        this.toast.error("Please provide template name and Excel file");
        return;
      }

      this.isUploadingTemplate = true;
      try {
        const formData = new FormData();
        formData.append("name", this.currentTemplate.name);
        formData.append("description", this.currentTemplate.description || "");
        formData.append("excel_file", this.currentTemplate.file);

        await axios.post(
          joinUrl(API_BASE_URL, "api/templates/upload"),
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );

        this.toast.success("Template uploaded successfully!");
        this.closeTemplateModals();
        await this.loadTemplates(false);
      } catch (error) {
        console.error("Failed to upload template:", error);
        this.toast.error(
          error.response?.data?.detail || "Failed to upload template"
        );
      } finally {
        this.isUploadingTemplate = false;
      }
    },

    editTemplate(template) {
      this.currentTemplate = {
        id: template.id,
        name: template.name,
        description: template.description || "",
        file: null,
      };
      this.showEditTemplateModal = true;
    },

    async updateTemplate() {
      if (!this.currentTemplate.name) {
        this.toast.error("Please provide template name");
        return;
      }

      this.isUploadingTemplate = true;
      try {
        const formData = new FormData();
        formData.append("name", this.currentTemplate.name);
        formData.append("description", this.currentTemplate.description || "");
        if (this.currentTemplate.file) {
          formData.append("excel_file", this.currentTemplate.file);
        }

        await axios.put(
          joinUrl(API_BASE_URL, `api/templates/${this.currentTemplate.id}`),
          formData,
          {
            headers: {
              "Content-Type": "multipart/form-data",
            },
          }
        );

        this.toast.success("Template updated successfully!");
        this.closeTemplateModals();
        await this.loadTemplates(false);
      } catch (error) {
        console.error("Failed to update template:", error);
        this.toast.error(
          error.response?.data?.detail || "Failed to update template"
        );
      } finally {
        this.isUploadingTemplate = false;
      }
    },

    async viewTemplateDetails(template) {
      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, `api/templates/${template.id}`)
        );
        this.viewingTemplate = response.data;
        // Create a deep copy of fields for editing
        this.editingFields = JSON.parse(
          JSON.stringify(this.viewingTemplate.template_data.fields)
        );
        this.showViewTemplateModal = true;
      } catch (error) {
        console.error("Failed to load template details:", error);
        this.toast.error("Failed to load template details");
      }
    },

    addNewField() {
      this.editingFields.push({
        name: "",
        description: "",
        location: "",
        output_format: "",
        priority: false,
      });

      // Scroll to bottom of fields container after DOM update
      this.$nextTick(() => {
        if (this.$refs.fieldsContainer) {
          this.$refs.fieldsContainer.scrollTo({
            top: this.$refs.fieldsContainer.scrollHeight,
            behavior: "smooth",
          });
        }
      });
    },

    deleteField(index) {
      if (confirm("Are you sure you want to delete this field?")) {
        this.editingFields.splice(index, 1);
      }
    },

    async saveFieldEdits() {
      if (!this.canSaveFields) {
        this.toast.error("All fields must have a name");
        return;
      }

      if (
        !this.viewingTemplate.name ||
        this.viewingTemplate.name.trim() === ""
      ) {
        this.toast.error("Template name is required");
        return;
      }

      this.isSavingFields = true;
      try {
        // First update the template name
        await axios.put(
          joinUrl(API_BASE_URL, `api/templates/${this.viewingTemplate.id}`),
          {
            name: this.viewingTemplate.name,
            description: this.viewingTemplate.description || "",
          },
          {
            headers: {
              "Content-Type": "application/json",
            },
          }
        );

        // Then update the fields
        const templateData = {
          fields: this.editingFields,
        };

        await axios.put(
          joinUrl(
            API_BASE_URL,
            `api/templates/${this.viewingTemplate.id}/fields`
          ),
          {
            template_data: templateData,
          },
          {
            headers: {
              "Content-Type": "application/json",
            },
          }
        );

        this.toast.success("Template updated successfully!");
        this.closeTemplateModals();
        await this.loadTemplates(false);
      } catch (error) {
        console.error("Failed to save field edits:", error);
        this.toast.error(
          error.response?.data?.detail || "Failed to save field edits"
        );
      } finally {
        this.isSavingFields = false;
      }
    },

    async exportTemplate(templateId, templateName) {
      try {
        const response = await axios.post(
          joinUrl(API_BASE_URL, `api/templates/${templateId}/export`),
          {},
          {
            responseType: "blob",
          }
        );

        // Create download link
        const url = window.URL.createObjectURL(new Blob([response.data]));
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute("download", `${templateName}.xlsx`);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);

        this.toast.success("Template exported successfully!");
      } catch (error) {
        console.error("Failed to export template:", error);
        this.toast.error("Failed to export template");
      }
    },

    async deleteTemplateConfirm(templateId, templateName) {
      if (
        !confirm(`Are you sure you want to delete template "${templateName}"?`)
      ) {
        return;
      }

      try {
        await axios.delete(
          joinUrl(API_BASE_URL, `api/templates/${templateId}`)
        );
        this.toast.success("Template deleted successfully!");
        await this.loadTemplates(false);
      } catch (error) {
        console.error("Failed to delete template:", error);
        this.toast.error("Failed to delete template");
      }
    },

    closeTemplateModals() {
      this.showAddTemplateModal = false;
      this.showEditTemplateModal = false;
      this.showViewTemplateModal = false;
      this.currentTemplate = {
        id: null,
        name: "",
        description: "",
        file: null,
      };
      this.viewingTemplate = null;
      this.editingFields = [];
      this.isSavingFields = false;
    },

    // ========================================================================
    // Prediction Instructions Manager Methods
    // ========================================================================

    async loadPredictionInstructions(resetPage = true) {
      if (resetPage) {
        this.activeTab = "prediction-instructions";
        this.predictionInstructionCurrentPage = 1;
      }

      this.predictionInstructionsLoading = true;
      try {
        const params = {
          page: this.predictionInstructionCurrentPage,
          page_size: this.predictionInstructionPageSize,
        };

        if (this.predictionInstructionTypeFilter) {
          params.instruction_type = this.predictionInstructionTypeFilter;
        }

        if (this.predictionInstructionSearch) {
          params.search = this.predictionInstructionSearch;
        }

        const response = await axios.get(
          joinUrl(API_BASE_URL, "api/prediction-instructions"),
          { params }
        );

        this.predictionInstructions = response.data.instructions;
        this.totalPredictionInstructions = response.data.total;
        this.predictionInstructionTotalPages = response.data.total_pages;
      } catch (error) {
        console.error("Failed to load prediction instructions:", error);
        this.toast.error("Failed to load instruction templates");
      } finally {
        this.predictionInstructionsLoading = false;
      }
    },

    onPredictionInstructionSearchChange() {
      if (this.predictionInstructionSearchTimeout) {
        clearTimeout(this.predictionInstructionSearchTimeout);
      }

      this.predictionInstructionSearchTimeout = setTimeout(() => {
        this.predictionInstructionCurrentPage = 1;
        this.loadPredictionInstructions(false);
      }, 500);
    },

    goToPredictionInstructionPage(page) {
      if (page < 1 || page > this.predictionInstructionTotalPages) return;
      this.predictionInstructionCurrentPage = page;
      this.loadPredictionInstructions(false);
    },

    changePredictionInstructionPageSize(newSize) {
      this.predictionInstructionPageSize = newSize;
      this.predictionInstructionCurrentPage = 1;
      this.loadPredictionInstructions(false);
    },

    async savePredictionInstruction() {
      if (!this.canSavePredictionInstruction) {
        this.toast.error("Please fill in all required fields");
        return;
      }

      this.isSavingPredictionInstruction = true;
      try {
        await axios.post(joinUrl(API_BASE_URL, "api/prediction-instructions"), {
          name: this.currentPredictionInstruction.name,
          instruction_type: this.currentPredictionInstruction.instruction_type,
          instructions_text:
            this.currentPredictionInstruction.instructions_text,
          description: this.currentPredictionInstruction.description || "",
        });

        this.toast.success("Instruction template created successfully!");
        this.closePredictionInstructionModals();
        await this.loadPredictionInstructions(false);
      } catch (error) {
        console.error("Failed to create instruction template:", error);
        this.toast.error(
          error.response?.data?.detail ||
            "Failed to create instruction template"
        );
      } finally {
        this.isSavingPredictionInstruction = false;
      }
    },

    async viewPredictionInstructionDetails(instruction) {
      try {
        const response = await axios.get(
          joinUrl(API_BASE_URL, `api/prediction-instructions/${instruction.id}`)
        );
        this.currentPredictionInstruction = {
          id: response.data.id,
          name: response.data.name,
          description: response.data.description || "",
          instruction_type: response.data.instruction_type,
          instructions_text: response.data.instructions_text,
        };
        this.showViewPredictionInstructionModal = true;
      } catch (error) {
        console.error("Failed to load instruction details:", error);
        this.toast.error("Failed to load instruction details");
      }
    },

    async updatePredictionInstruction() {
      if (!this.canSavePredictionInstruction) {
        this.toast.error("Please fill in all required fields");
        return;
      }

      this.isSavingPredictionInstruction = true;
      try {
        await axios.put(
          joinUrl(
            API_BASE_URL,
            `api/prediction-instructions/${this.currentPredictionInstruction.id}`
          ),
          {
            name: this.currentPredictionInstruction.name,
            description: this.currentPredictionInstruction.description || "",
            instructions_text:
              this.currentPredictionInstruction.instructions_text,
          }
        );

        this.toast.success("Instruction template updated successfully!");
        this.closePredictionInstructionModals();
        await this.loadPredictionInstructions(false);
      } catch (error) {
        console.error("Failed to update instruction template:", error);
        this.toast.error(
          error.response?.data?.detail ||
            "Failed to update instruction template"
        );
      } finally {
        this.isSavingPredictionInstruction = false;
      }
    },

    async deletePredictionInstructionConfirm(instructionId, instructionName) {
      if (
        !confirm(
          `Are you sure you want to delete instruction template "${instructionName}"?`
        )
      ) {
        return;
      }

      try {
        await axios.delete(
          joinUrl(API_BASE_URL, `api/prediction-instructions/${instructionId}`)
        );
        this.toast.success("Instruction template deleted successfully!");
        await this.loadPredictionInstructions(false);
      } catch (error) {
        console.error("Failed to delete instruction template:", error);
        this.toast.error("Failed to delete instruction template");
      }
    },

    closePredictionInstructionModals() {
      this.showAddPredictionInstructionModal = false;
      this.showViewPredictionInstructionModal = false;
      this.currentPredictionInstruction = {
        id: null,
        name: "",
        description: "",
        instruction_type: "",
        instructions_text: "",
      };
      this.isSavingPredictionInstruction = false;
    },

    formatDate(dateString) {
      if (!dateString) return "N/A";
      const date = new Date(dateString);
      return date.toLocaleDateString() + " " + date.toLocaleTimeString();
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
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 3rem;
  flex-wrap: wrap;
  padding: 0 1rem;
}

.tab-btn {
  background: white;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  padding: 0.85rem 1.5rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: #64748b;
  cursor: pointer;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  white-space: nowrap;
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

/* Dropdown Navigation Styles */
.dropdown-container {
  position: relative;
}

.dropdown-btn {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.dropdown-arrow {
  font-size: 0.7rem;
  transition: transform 0.3s ease;
}

.dropdown-menu {
  position: absolute;
  top: calc(100% + 0.5rem);
  left: 0;
  min-width: 220px;
  background: white;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  overflow: hidden;
  animation: dropdownSlideDown 0.2s ease;
}

@keyframes dropdownSlideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.dropdown-item {
  width: 100%;
  padding: 0.9rem 1.2rem;
  border: none;
  background: white;
  text-align: left;
  font-size: 0.95rem;
  font-weight: 500;
  color: #64748b;
  cursor: pointer;
  transition: all 0.2s ease;
  border-bottom: 1px solid #f1f5f9;
}

.dropdown-item:last-child {
  border-bottom: none;
}

.dropdown-item:hover {
  background: #f8fafc;
  color: #3b82f6;
  padding-left: 1.5rem;
}

.dropdown-item:active {
  background: #eff6ff;
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

.download-format-group {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
}

.download-btn-alt {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.3);
}

.download-btn-alt:hover {
  transform: translateY(-2px);
  box-shadow: 0 10px 25px -3px rgba(16, 185, 129, 0.4);
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

/* Password Unlock Section */
.password-unlock-section {
  display: flex;
  justify-content: center;
  margin: 2rem 0;
}

.password-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 16px;
  padding: 3rem;
  text-align: center;
  max-width: 600px;
  width: 100%;
  box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
}

.password-icon {
  font-size: 4rem;
  margin-bottom: 1.5rem;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%,
  100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.1);
  }
}

.password-card h3 {
  font-size: 1.8rem;
  font-weight: 700;
  color: white;
  margin-bottom: 1rem;
}

.password-card > p {
  color: rgba(255, 255, 255, 0.9);
  font-size: 1rem;
  line-height: 1.6;
  margin-bottom: 2rem;
}

.password-input-group {
  display: flex;
  gap: 1rem;
  margin-bottom: 2rem;
}

.password-input {
  flex: 1;
  padding: 1rem;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-radius: 12px;
  font-size: 1rem;
  background: rgba(255, 255, 255, 0.1);
  color: white;
  transition: all 0.3s ease;
}

.password-input::placeholder {
  color: rgba(255, 255, 255, 0.6);
}

.password-input:focus {
  outline: none;
  border-color: white;
  background: rgba(255, 255, 255, 0.2);
}

.unlock-btn {
  padding: 1rem 2rem;
  background: white;
  color: #667eea;
  border: none;
  border-radius: 12px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  white-space: nowrap;
}

.unlock-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
}

.unlock-btn:active {
  transform: translateY(0);
}

.password-features {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.password-features .feature-item {
  background: rgba(255, 255, 255, 0.15);
  border: 1px solid rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(10px);
}

.password-features .feature-item span:last-child {
  color: white;
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

/* ========================================================================
   Modifiers Config Styles
   ======================================================================== */

.modifiers-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  gap: 1rem;
}

.search-box {
  flex: 1;
  max-width: 400px;
}

.search-input {
  width: 100%;
  padding: 0.75rem 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  transition: all 0.3s ease;
}

.search-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.add-btn {
  padding: 0.75rem 1.5rem;
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
}

.add-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
}

.bulk-import-section {
  margin-bottom: 2rem;
}

.bulk-import-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 2rem;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
}

.bulk-import-card h3 {
  margin: 0 0 0.5rem 0;
  font-size: 1.5rem;
  font-weight: 700;
}

.bulk-import-card p {
  margin: 0 0 1.5rem 0;
  opacity: 0.9;
}

.csv-format-info {
  background: rgba(255, 255, 255, 0.1);
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1.5rem;
  border-left: 4px solid #fbbf24;
}

.csv-format-info strong {
  display: block;
  margin-bottom: 0.5rem;
  font-size: 1rem;
}

.format-example {
  background: rgba(0, 0, 0, 0.2);
  padding: 0.75rem;
  border-radius: 6px;
  margin: 0.75rem 0;
  font-family: "Courier New", monospace;
}

.format-example code {
  color: #fbbf24;
  font-size: 0.875rem;
  line-height: 1.6;
}

.format-note {
  margin: 0.5rem 0 0 0 !important;
  font-size: 0.875rem;
  opacity: 0.9;
}

.format-note strong {
  display: inline;
  color: #fbbf24;
  font-weight: 700;
}

.bulk-import-controls {
  display: flex;
  gap: 1rem;
  align-items: center;
  flex-wrap: wrap;
}

.upload-btn {
  background: white;
  color: #667eea;
  padding: 0.75rem 1.5rem;
  border: none;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.upload-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3);
}

.upload-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.import-btn {
  background: #10b981;
  color: white;
  padding: 0.75rem 1.5rem;
  border: 2px solid white;
  border-radius: 8px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.import-btn:hover {
  background: #059669;
  transform: translateY(-2px);
}

.import-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.file-name {
  background: rgba(255, 255, 255, 0.2);
  padding: 0.5rem 1rem;
  border-radius: 6px;
  font-size: 0.875rem;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
  user-select: none;
}

.checkbox-label input[type="checkbox"] {
  width: 18px;
  height: 18px;
  cursor: pointer;
}

.import-result {
  margin-top: 1.5rem;
  padding: 1rem;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.1);
}

.import-result .success-message {
  color: white;
}

.import-result .success-message ul {
  margin: 0.5rem 0 0 0;
  padding-left: 1.5rem;
}

.import-result .success-message li {
  margin: 0.25rem 0;
}

.import-result .error-message {
  color: #fecaca;
  font-weight: 600;
}

.mappings-editor {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.mapping-row {
  display: flex;
  gap: 1rem;
  align-items: center;
}

.mapping-row .form-input {
  flex: 1;
}

.mapping-row .delete-btn {
  flex-shrink: 0;
}

.add-btn.small,
.delete-btn.small {
  padding: 0.5rem 1rem;
  font-size: 0.875rem;
}

.large-modal {
  max-width: 900px;
  max-height: 80vh;
  overflow-y: auto;
}

.mappings-list {
  max-height: 400px;
  overflow-y: auto;
  margin-top: 1rem;
}

.template-info {
  background: #f3f4f6;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1rem;
}

.template-info p {
  margin: 0.5rem 0;
}

.toggle-selector {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
  background: #f3f4f6;
  padding: 0.25rem;
  border-radius: 8px;
}

.toggle-btn {
  flex: 1;
  padding: 0.75rem;
  border: none;
  background: transparent;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.3s ease;
}

.toggle-btn.active {
  background: white;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  color: #3b82f6;
}

.toggle-btn:hover:not(.active) {
  background: rgba(255, 255, 255, 0.5);
}

.template-selector {
  display: flex;
  gap: 0.5rem;
  align-items: center;
}

.template-select {
  flex: 1;
  padding: 0.75rem;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  font-size: 1rem;
  background: white;
  cursor: pointer;
}

.template-select:focus {
  outline: none;
  border-color: #3b82f6;
}

.refresh-btn {
  padding: 0.75rem 1rem;
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  transition: all 0.3s ease;
}

.refresh-btn:hover {
  background: #2563eb;
  transform: scale(1.05);
}

.field-instructions-item {
  background: #f9fafb;
  padding: 1rem;
  border-radius: 8px;
  margin-bottom: 1rem;
  border-left: 4px solid #3b82f6;
}

.field-instructions-item h4 {
  margin: 0 0 0.75rem 0;
  color: #1f2937;
  font-size: 1.1rem;
}

.field-instructions-item ul {
  margin: 0;
  padding-left: 1.5rem;
}

.field-instructions-item li {
  margin: 0.5rem 0;
  color: #4b5563;
  line-height: 1.5;
}

.modifiers-table-container {
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.modifiers-table {
  width: 100%;
  border-collapse: collapse;
}

.modifiers-table thead {
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  color: white;
}

.modifiers-table th {
  padding: 1rem;
  text-align: left;
  font-weight: 600;
  font-size: 0.875rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.modifiers-table tbody tr {
  border-bottom: 1px solid #e2e8f0;
  transition: background-color 0.2s ease;
}

.modifiers-table tbody tr:hover {
  background-color: #f8fafc;
}

.modifiers-table td {
  padding: 1rem;
  font-size: 0.938rem;
}

.badge-yes {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  background-color: #d1fae5;
  color: #065f46;
  border-radius: 9999px;
  font-size: 0.813rem;
  font-weight: 600;
}

.badge-no {
  display: inline-block;
  padding: 0.25rem 0.75rem;
  background-color: #fee2e2;
  color: #991b1b;
  border-radius: 9999px;
  font-size: 0.813rem;
  font-weight: 600;
}

.action-btn {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  margin-right: 0.5rem;
}

.edit-btn {
  background-color: #dbeafe;
  color: #1e40af;
}

.edit-btn:hover {
  background-color: #bfdbfe;
  transform: translateY(-1px);
}

.delete-btn {
  background-color: #fee2e2;
  color: #991b1b;
}

.delete-btn:hover {
  background-color: #fecaca;
  transform: translateY(-1px);
}

.pagination-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background-color: #f8fafc;
  border-top: 1px solid #e2e8f0;
  flex-wrap: wrap;
  gap: 1rem;
}

.pagination-info {
  color: #64748b;
  font-size: 0.875rem;
  font-weight: 500;
}

.pagination-controls {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.pagination-btn {
  padding: 0.5rem 1rem;
  background: white;
  border: 2px solid #e2e8f0;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  color: #475569;
}

.pagination-btn:hover:not(:disabled) {
  border-color: #3b82f6;
  background-color: #eff6ff;
  color: #3b82f6;
}

.pagination-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.page-info {
  padding: 0 1rem;
  font-weight: 600;
  color: #1e293b;
  font-size: 0.938rem;
}

.page-size-selector {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.page-size-selector label {
  font-size: 0.875rem;
  color: #64748b;
  font-weight: 500;
}

.page-size-select {
  padding: 0.5rem;
  border: 2px solid #e2e8f0;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  background: white;
  color: #475569;
  transition: all 0.2s ease;
}

.page-size-select:hover {
  border-color: #3b82f6;
}

.page-size-select:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.loading-section {
  text-align: center;
  padding: 3rem;
}

.spinner {
  border: 4px solid #e2e8f0;
  border-top: 4px solid #3b82f6;
  border-radius: 50%;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

.empty-state {
  text-align: center;
  padding: 3rem;
  color: #64748b;
}

/* Modal Styles */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: white;
  border-radius: 12px;
  width: 90%;
  max-width: 500px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
  animation: modalSlideIn 0.3s ease;
}

@keyframes modalSlideIn {
  from {
    opacity: 0;
    transform: translateY(-20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem;
  border-bottom: 1px solid #e2e8f0;
}

.modal-header h3 {
  margin: 0;
  font-size: 1.5rem;
  color: #1e293b;
}

.close-btn {
  background: none;
  border: none;
  font-size: 1.5rem;
  cursor: pointer;
  color: #64748b;
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  transition: all 0.2s ease;
}

.close-btn:hover {
  background-color: #f1f5f9;
  color: #1e293b;
}

.modal-body {
  padding: 1.5rem;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  margin-bottom: 0.5rem;
  font-weight: 600;
  color: #334155;
  font-size: 0.938rem;
}

.form-input {
  width: 100%;
  padding: 0.75rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  transition: all 0.3s ease;
}

.form-input:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.form-input:disabled {
  background-color: #f1f5f9;
  cursor: not-allowed;
}

.form-checkbox {
  width: 18px;
  height: 18px;
  margin-right: 0.5rem;
  cursor: pointer;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 1rem;
  padding: 1.5rem;
  border-top: 1px solid #e2e8f0;
}

.btn-primary {
  padding: 0.75rem 1.5rem;
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-primary:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-secondary {
  padding: 0.75rem 1.5rem;
  background: white;
  color: #64748b;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-secondary:hover {
  border-color: #cbd5e1;
  background-color: #f8fafc;
}

/* ========================================================================
   Templates Manager Styles
   ======================================================================== */

/* Templates Grid Layout */
.templates-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}

/* Template Card */
.template-card {
  background: white;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  padding: 1.5rem;
  transition: all 0.3s ease;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.template-card:hover {
  border-color: #3b82f6;
  box-shadow: 0 8px 16px rgba(59, 130, 246, 0.15);
  transform: translateY(-2px);
}

.template-card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 1rem;
  gap: 1rem;
}

.template-card-header h3 {
  font-size: 1.1rem;
  font-weight: 600;
  color: #1e293b;
  flex: 1;
  margin: 0;
  line-height: 1.4;
}

.template-card-actions {
  display: flex;
  gap: 0.5rem;
  flex-shrink: 0;
}

.template-card-body {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.template-description {
  color: #64748b;
  font-size: 0.9rem;
  line-height: 1.5;
  margin: 0;
}

.template-meta {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.8rem;
  color: #94a3b8;
}

.template-date {
  display: block;
}

/* Template Actions */
.action-btn {
  padding: 0.4rem 0.7rem;
  border: none;
  border-radius: 6px;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s ease;
  background-color: #f1f5f9;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #475569;
}

.action-btn svg {
  display: block;
}

.action-btn:hover {
  transform: scale(1.05);
}

.view-btn:hover {
  background-color: #dbeafe;
}

.edit-btn:hover {
  background-color: #fef3c7;
}

.action-btn.download-btn {
  background: #f1f5f9 !important;
  background-color: #f1f5f9 !important;
  background-image: none !important;
  color: #475569 !important;
  box-shadow: none !important;
  padding: 0.4rem 0.7rem !important;
}

.action-btn.download-btn:hover {
  background: #e2e8f0 !important;
  background-color: #e2e8f0 !important;
  background-image: none !important;
  transform: scale(1.05) !important;
  box-shadow: none !important;
}

.delete-btn {
  background-color: #fee2e2;
  color: #dc2626;
}

.delete-btn:hover {
  background-color: #fecaca;
}

/* Template Selection in Processing Tabs */
.template-selection-toggle {
  margin: 1rem 0;
  padding: 0.75rem;
  background: #f8fafc;
  border-radius: 8px;
}

.toggle-label {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  cursor: pointer;
  font-size: 0.95rem;
  color: #475569;
}

.toggle-checkbox {
  width: 1.25rem;
  height: 1.25rem;
  cursor: pointer;
}

.toggle-text {
  font-weight: 500;
}

.template-dropdown-section {
  padding: 1rem;
  background: #f1f5f9;
  border-radius: 8px;
}

.template-select {
  width: 100%;
  padding: 0.75rem 1rem;
  font-size: 1rem;
  border: 2px solid #cbd5e1;
  border-radius: 8px;
  background: white;
  color: #1e293b;
  cursor: pointer;
  transition: all 0.3s ease;
}

.template-select:hover {
  border-color: #3b82f6;
}

.template-select:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.template-hint {
  margin-top: 0.75rem;
  font-size: 0.85rem;
  color: #64748b;
  text-align: center;
}

.manage-link {
  color: #3b82f6;
  text-decoration: none;
  font-weight: 500;
}

.manage-link:hover {
  text-decoration: underline;
}

/* Template Details Modal */
.modal-large {
  max-width: 900px;
  max-height: 80vh;
  overflow-y: auto;
}

.template-detail-section {
  margin-bottom: 2rem;
}

.template-detail-section h4 {
  font-size: 1.3rem;
  color: #1e293b;
  margin-bottom: 0.5rem;
}

.template-fields-section {
  margin-top: 2rem;
}

.template-fields-section h4 {
  font-size: 1.1rem;
  color: #1e293b;
  margin-bottom: 1rem;
}

.fields-table-container {
  overflow-x: auto;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}

.fields-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

.fields-table thead {
  background-color: #f8fafc;
}

.fields-table th {
  padding: 0.75rem;
  text-align: left;
  font-weight: 600;
  color: #475569;
  border-bottom: 2px solid #e2e8f0;
}

.fields-table td {
  padding: 0.75rem;
  border-bottom: 1px solid #f1f5f9;
  color: #64748b;
}

.fields-table tr:hover {
  background-color: #f8fafc;
}

/* Form Elements for Template Modal */
.form-textarea {
  width: 100%;
  padding: 0.75rem;
  border: 2px solid #e2e8f0;
  border-radius: 8px;
  font-family: inherit;
  font-size: 1rem;
  resize: vertical;
  transition: border-color 0.3s ease;
}

.form-textarea:focus {
  outline: none;
  border-color: #3b82f6;
}

.form-file-input {
  width: 100%;
  padding: 0.75rem;
  border: 2px dashed #cbd5e1;
  border-radius: 8px;
  background: #f8fafc;
  cursor: pointer;
  transition: all 0.3s ease;
}

.form-file-input:hover {
  border-color: #3b82f6;
  background: #eff6ff;
}

.file-selected {
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: #dbeafe;
  border-radius: 6px;
  font-size: 0.9rem;
  color: #1e40af;
}

.file-hint {
  margin-top: 0.5rem;
  font-size: 0.85rem;
  color: #94a3b8;
  font-style: italic;
}

/* Field Editor Styles */
.fields-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
}

.fields-header h4 {
  margin: 0;
}

.add-field-btn {
  padding: 0.6rem 1.2rem;
  background: #10b981;
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
}

.add-field-btn:hover {
  background: #059669;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
}

.fields-editor-container {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
  max-height: 500px;
  overflow-y: auto;
  padding-right: 0.5rem;
}

.field-editor-card {
  background: #f8fafc;
  border: 2px solid #e2e8f0;
  border-radius: 12px;
  padding: 1.5rem;
  transition: all 0.3s ease;
}

.field-editor-card:hover {
  border-color: #cbd5e1;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.field-editor-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.field-number {
  font-size: 0.9rem;
  font-weight: 600;
  color: #64748b;
  padding: 0.25rem 0.75rem;
  background: white;
  border-radius: 6px;
}

.delete-field-btn {
  padding: 0.4rem 0.8rem;
  background: #fee2e2;
  color: #dc2626;
  border: none;
  border-radius: 6px;
  font-size: 0.9rem;
  cursor: pointer;
  transition: all 0.2s ease;
}

.delete-field-btn:hover {
  background: #fecaca;
  transform: scale(1.05);
}

.field-editor-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

.field-editor-grid .form-group.full-width {
  grid-column: 1 / -1;
}

.field-editor-grid .form-group.priority-group {
  grid-column: 1 / -1;
  margin-top: 0.5rem;
}

.checkbox-label {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  cursor: pointer;
  padding: 0.75rem;
  background: white;
  border-radius: 8px;
  border: 2px solid #e2e8f0;
  transition: all 0.3s ease;
}

.checkbox-label:hover {
  border-color: #3b82f6;
  background: #eff6ff;
}

.checkbox-label input[type="checkbox"] {
  width: 1.25rem;
  height: 1.25rem;
  cursor: pointer;
}

.checkbox-label span {
  font-size: 0.95rem;
  color: #475569;
  font-weight: 500;
}

/* Prediction Instructions Styles */
.filter-group {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  background: white;
  border-radius: 8px;
  border: 2px solid #e2e8f0;
}

.filter-group label {
  font-size: 0.9rem;
  font-weight: 600;
  color: #475569;
}

.type-filter-select {
  padding: 0.5rem 1rem;
  border: 2px solid #e2e8f0;
  border-radius: 6px;
  font-size: 0.9rem;
  background: white;
  cursor: pointer;
  transition: all 0.3s ease;
}

.type-filter-select:hover {
  border-color: #3b82f6;
}

.type-filter-select:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.template-title-section {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex: 1;
}

.instruction-type-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 6px;
  font-size: 0.75rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.badge-cpt {
  background: #dbeafe;
  color: #1e40af;
}

.badge-icd {
  background: #fef3c7;
  color: #92400e;
}

.instruction-textarea {
  font-family: "Monaco", "Menlo", "Consolas", monospace;
  font-size: 0.9rem;
  line-height: 1.6;
}
</style>
