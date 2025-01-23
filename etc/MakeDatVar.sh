#!/bin/bash
# The next line is executed by /bin/bash, but not Tcl \
# exec tclsh "$0" "$@"

# Copyright (c) 2019, Parallax Software, Inc.
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Usage: MakeDatVar var_name var_file dat_file
# var_name powv9 
# var_file ${POWV9_CPP} 
# dat_file ${POWV9_DAT}

# Check if we have 3 arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 var_name var_file dat_file"
    exit 1
fi

# Get command line arguments
var="$1"
var_file="$2"
dat_file="$3"

# Create initial C++ file with header
cat > "$var_file" << EOF
#include <string>
namespace Flute {
std::string $var = "
EOF

# Create temporary files
b64_file="${dat_file%.*}.b64"
b64_file2="${dat_file%.*}.tr"

# Convert data file to base64 and strip newlines
base64 -i "$dat_file" > "$b64_file"
tr -d '\n' < "$b64_file" > "$b64_file2"
cat "$b64_file2" >> "$var_file"

# Add closing syntax
cat >> "$var_file" << EOF
";
}
EOF

# Clean up temporary files
rm -f "$b64_file" "$b64_file2"