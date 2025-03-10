<template>
  <div class="dataset-upload-container">
    <h2>Upload Dataset</h2>

    <FileUpload
        ref="fileUploadRef"
        mode="basic"
        name="datasetFile"
        :customUpload="true"
        @uploader="handleCustomUpload"
        accept=".csv"
        :maxFileSize="10000000"
        chooseLabel="Choose CSV File"
        class="p-mt-2"
        :auto="false"  
    />

    <Dialog
        v-model:visible="previewVisible"
        :modal="true"
        header="File Preview"
        :style="{ width: '75vw' }"
        :maximizable="true"
    >
      <div v-if="previewFile">
          <!--  Можно добавить превью содержимого файла (например, первые строки), но это требует доп. логики на клиенте.  -->
        <p><strong>Filename:</strong> {{ previewFile.name }}</p>
        <p><strong>Size:</strong> {{ formatFileSize(previewFile.size) }}</p>
          <div  v-if="previewContent" style="overflow-x: auto;"> <!-- Добавлено для прокрутки -->
             <pre>{{ previewContent }}</pre>  <!--  Отображаем первые строки, как текст -->
          </div>
          <div v-else>
            Loading preview...
            <ProgressSpinner />
          </div>
      </div>
      <template #footer>
        <Button label="Cancel" icon="pi pi-times" @click="cancelUpload" class="p-button-text" />
        <Button label="Upload" icon="pi pi-upload" @click="confirmUpload" :disabled="uploading" :loading="uploading"/>
      </template>
    </Dialog>

     <Toast />
  </div>
</template>

<script>
import FileUpload from 'primevue/fileupload';
import Dialog from 'primevue/dialog';
import Button from 'primevue/button';
import Toast from 'primevue/toast';
import ProgressSpinner from 'primevue/progressspinner'; // Добавили
import axios from 'axios';
import { parse } from 'papaparse';   // Установи: npm install papaparse


const API_BASE_URL = 'http://localhost:8000';

export default {
  name: 'DatasetUpload',
  components: {
    FileUpload,
    Dialog,
    Button,
    Toast,
    ProgressSpinner,    
  },
  data() {
    return {
      previewVisible: false,
      previewFile: null,
        previewContent: null,    //  Для хранения содержимого превью
      uploading: false,
      errorMessage: null, //Сообщение ошибки
    };
  },
  methods: {
      async handleCustomUpload(event) {  //  Обработчик для customUpload
           this.previewFile = event.files[0];  //  Сохраняем файл для превью
          if (!this.previewFile) return;          
        this.previewVisible = true;         //  Показываем модальное окно
          this.previewContent = null; // Сброс
          this.loadPreviewContent();

      },

      async loadPreviewContent() {
            //  Загрузка превью содержимого файла (первые N строк)
            try {
                const text = await this.previewFile.text();  // Читаем как текст
                 parse(text, {
                    preview: 10,  //  Показываем первые 10 строк
                    complete: (results) => {
                        // Преобразуем в строку для отображения в <pre>
                        this.previewContent = results.data.map(row => row.join(',')).join('\n');
                    },
                    error: (error) => {
                         console.error("CSV parsing error:", error);
                         this.previewContent = "Error loading preview.";
                    }
                });
            } catch (error) {
                console.error("Error reading file:", error);
                this.previewContent = "Error loading preview.";
             }
      },
      async confirmUpload() {        
        if (!this.previewFile) return;
          this.uploading = true;          //  Показываем, что идет загрузка          
           this.previewVisible = false;  //Закрываем окно (до отправки на сервер, чтобы не блокировать)
        const formData = new FormData();
        formData.append('file', this.previewFile);

        try {
          const response = await axios.post(`${API_BASE_URL}/upload_dataset/`, formData, {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          });
          
          this.$toast.add({ severity: 'success', summary: 'Success', detail: 'Dataset uploaded successfully!', life: 3000 });          
          this.$emit('dataset-uploaded', { filename: response.data.filename, columns: response.data.columns });
          this.resetUpload();  //  Сбрасываем состояние
        } catch (error) {
             let detail = "An error occurred during file upload.";  // Default
             if (error.response) {                
                detail = error.response.data.detail || detail; //  Используем detail из ответа, если есть
             }            
            this.$toast.add({ severity: 'error', summary: 'Upload Failed', detail: detail, life: 5000 });           
          console.error("File upload failed:", error);            
        } finally {
          this.uploading = false;
        }
      },
    cancelUpload() {
    this.resetUpload();
      this.$refs.fileUploadRef.clear(); // Очистка FileUpload
      this.previewVisible = false;      
      this.previewFile = null;      
    },

    resetUpload() {    
    this.previewFile = null;
      this.previewContent = null;
      this.uploading = false;
      this.errorMessage = null;
      if (this.$refs.fileUploadRef) {
           this.$refs.fileUploadRef.clear(); // Добавил очистку
      }       
    },
     formatFileSize(bytes) {
      if (bytes === 0) return '0 Bytes';
      const k = 1024;
      const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
  }
};
</script>

<style scoped>
.dataset-upload-container {
  padding: 20px;
}

/* Стили для модального окна, если нужно */
.p-dialog .p-dialog-content {
    overflow-y: auto; /* Добавлено для прокрутки содержимого, если не помещается */    
}
</style>
