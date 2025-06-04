# ContextualOpenAI

If you want to make your OpenAI generate context-based responses and copy the necessary files in this repository, follow these simple steps
1. Log in to the repository where you want to copy the files and enable context-based responses 
2. Copy and paste the following script:
  git clone --depth 1 --filter=blob:none --sparse https://github.com/gan0412/ContextualOpenAI.git temp_ContextualOpenAI
  cd temp_ContextualOpenAI
  git sparse-checkout set context.py chistory.py
  mv context.py chistory.py ..
  cd ..
  rm -rf temp_ContextualOpenAI
  echo "Files have been copied to the current directory."
