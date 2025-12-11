#!/bin/bash

# Script to copy all Sea of Nodes chapters to the docs directory

SOURCE_DIR="/tmp/sea-of-nodes-docs"
TARGET_DIR="/Users/jim/work/cppfort/docs/sea-of-nodes"

echo "Copying Sea of Nodes documentation..."
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"

# Copy README
cp "$SOURCE_DIR/README.md" "$TARGET_DIR/README.md"

# Copy all chapter directories
for i in {01..24}; do
    if [ -d "$SOURCE_DIR/chapter$i" ]; then
        echo "Copying chapter$i..."
        cp -r "$SOURCE_DIR/chapter$i" "$TARGET_DIR/"
    fi
done

# Copy the TypeAnalysis.md file
if [ -f "$SOURCE_DIR/TypeAnalysis.md" ]; then
    echo "Copying TypeAnalysis.md..."
    cp "$SOURCE_DIR/TypeAnalysis.md" "$TARGET_DIR/"
fi

# Copy ASimpleReply.md
if [ -f "$SOURCE_DIR/ASimpleReply.md" ]; then
    echo "Copying ASimpleReply.md..."
    cp "$SOURCE_DIR/ASimpleReply.md" "$TARGET_DIR/"
fi

# Copy the docs directory with diagrams
if [ -d "$SOURCE_DIR/docs" ]; then
    echo "Copying docs directory with diagrams..."
    cp -r "$SOURCE_DIR/docs" "$TARGET_DIR/"
fi

echo "Done copying Sea of Nodes documentation!"

# List what was copied
echo ""
echo "Copied files:"
ls -la "$TARGET_DIR/"

echo ""
echo "Chapter directories:"
ls -1 "$TARGET_DIR/chapter"*.md 2>/dev/null || echo "Chapter README files found"