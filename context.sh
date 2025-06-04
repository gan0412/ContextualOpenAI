git clone --depth 1 --filter=blob:none --sparse https://github.com/gan0412/ContextualOpenAI.git temp_ContextualOpenAI
cd temp_ContextualOpenAI
git sparse-checkout set context.py chistory.py

# Move the files to the current directory
mv context.py chistory.py ..

# Go back to the parent directory and remove the temporary clone directory
cd ..
rm -rf temp_ContextualOpenAI

echo "Files have been copied to the current directory."