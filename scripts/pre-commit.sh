#!/bin/sh
#
# An example hook script to verify what is about to be committed.
# Called by "git commit" with no arguments.  The hook should
# exit with non-zero status after issuing an appropriate message if
# it wants to stop the commit.
#
# To enable this hook, rename this file to "pre-commit".

if git rev-parse --verify HEAD >/dev/null 2>&1
then
	against=HEAD
else
	# Initial commit: diff against an empty tree object
	against=$(git hash-object -t tree /dev/null)
fi


# Find notebooks to be committed
(
IFS='
'
NBS=`git diff-index --cached $against --name-only | grep -a '.ipynb$' | uniq`

for NB in $NBS ; do
    echo "Removing outputs from $NB"
    nbstripout "$NB"
    git add "$NB"
done
)


# Format py files

(
IFS='
'
PYS=`git diff-index --cached $against --name-only | grep -a '.py$' | uniq`

for PY in $PYS ; do
    echo "Formatting $PY"
    black "$PY"
    git add "$PY"
done
)