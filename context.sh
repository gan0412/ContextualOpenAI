git clone --depth 1 --filter=blob:none --sparse https://github.com/gan0412/ContextualOpenAI.git temp_ContextualOpenAI
cd temp_ContextualOpenAI
git sparse-checkout set context.py chistory.py
mv context.py chistory.py ..
cd ..
rm -rf temp_ContextualOpenAI
echo "Files have been copied to the current directory."