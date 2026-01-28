<template>
  <v-select
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    :options="options"
    :reduce="reduce"
    :label="label"
    :placeholder="placeholder"
    :clearable="clearable"
    :searchable="true"
    :filterable="true"
    class="searchable-template-select"
    @open="$emit('focus')"
  >
    <template #no-options>
      No options available
    </template>
  </v-select>
</template>

<script>
import vSelect from 'vue-select'
import 'vue-select/dist/vue-select.css'

export default {
  name: 'SearchableSelect',
  components: {
    vSelect
  },
  props: {
    modelValue: {
      type: [String, Number, Object],
      default: null
    },
    options: {
      type: Array,
      required: true
    },
    reduce: {
      type: Function,
      default: null
    },
    label: {
      type: String,
      default: 'label'
    },
    placeholder: {
      type: String,
      default: 'Select an option...'
    },
    clearable: {
      type: Boolean,
      default: true
    }
  },
  emits: ['update:modelValue', 'focus']
}
</script>

<style>
/* Override vue-select styles to match the existing template-select styling */
.searchable-template-select .vs__dropdown-toggle {
  width: 100%;
  padding: 0.5rem 0.75rem;
  font-size: 1rem;
  border: 2px solid #cbd5e1;
  border-radius: 8px;
  background: white;
  color: #1e293b;
  cursor: pointer;
  transition: all 0.3s ease;
  min-height: 48px;
}

.searchable-template-select .vs__dropdown-toggle:hover {
  border-color: #3b82f6;
}

.searchable-template-select.vs--open .vs__dropdown-toggle {
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.searchable-template-select .vs__selected {
  color: #1e293b;
  margin: 0;
  padding: 0;
  border: none;
}

.searchable-template-select .vs__search {
  margin: 0;
  padding: 0;
  border: none;
  color: #1e293b;
}

.searchable-template-select .vs__search::placeholder {
  color: #94a3b8;
}

.searchable-template-select .vs__actions {
  padding: 0 0.25rem;
}

.searchable-template-select .vs__clear {
  margin-right: 0.25rem;
}

.searchable-template-select .vs__open-indicator {
  fill: #64748b;
}

.searchable-template-select .vs__dropdown-menu {
  border: 2px solid #3b82f6;
  border-radius: 8px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  max-height: 300px;
  overflow-y: auto;
}

.searchable-template-select .vs__dropdown-option {
  padding: 0.75rem 1rem;
  color: #1e293b;
  transition: all 0.2s ease;
}

.searchable-template-select .vs__dropdown-option--highlight {
  background: #eff6ff;
  color: #1e40af;
}

.searchable-template-select .vs__dropdown-option--selected {
  background: #dbeafe;
  color: #1e40af;
  font-weight: 500;
}

.searchable-template-select .vs__no-options {
  padding: 0.75rem 1rem;
  color: #64748b;
  text-align: center;
}

/* Focus state */
.searchable-template-select.vs--single.vs--open .vs__selected {
  opacity: 0.4;
}

/* Make sure the component takes full width */
.searchable-template-select {
  width: 100%;
}
</style>
