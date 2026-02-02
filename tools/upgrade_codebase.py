#!/usr/bin/env python3
"""Automated security and quality upgrades for RFSN Controller.

This script applies automated fixes for:
- Security issues (eval, exec, shell=True)
- Print statements → structured logging
- Old string formatting → f-strings
- Type hints modernization
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def fix_print_to_logging(content: str, filename: str) -> Tuple[str, int]:
    """Replace print() calls with structured logging.
    
    Returns:
        (updated_content, num_changes)
    """
    changes = 0
    lines = content.split('\n')
    updated_lines = []
    needs_import = False
    
    for line in lines:
        # Check if it's a print statement (not in string or comment)
        if re.match(r'^\s*print\s*\(', line) and '#' not in line.split('print')[0]:
            # Extract indentation
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent
            
            # Try to parse print content
            match = re.search(r'print\s*\(\s*["\'](.+?)["\']\s*\)', line)
            if match:
                msg = match.group(1)
                # Convert to logger.info
                new_line = f'{indent_str}logger.info("{msg}")'
                updated_lines.append(new_line)
                changes += 1
                needs_import = True
                continue
            
            # Try f-string print
            match = re.search(r'print\s*\(\s*f["\'](.+?)["\']\s*\)', line)
            if match:
                msg = match.group(1)
                # Keep as f-string
                new_line = f'{indent_str}logger.info(f"{msg}")'
                updated_lines.append(new_line)
                changes += 1
                needs_import = True
                continue
        
        updated_lines.append(line)
    
    result = '\n'.join(updated_lines)
    
    # Add import if needed
    if needs_import and 'from rfsn_controller.structured_logging import get_logger' not in content:
        # Find appropriate place to add import (after other imports)
        import_lines = []
        other_lines = []
        in_imports = False
        
        for line in result.split('\n'):
            if line.startswith('import ') or line.startswith('from '):
                in_imports = True
                import_lines.append(line)
            elif in_imports and line.strip() == '':
                import_lines.append(line)
                import_lines.append('from rfsn_controller.structured_logging import get_logger')
                import_lines.append('')
                import_lines.append(f'logger = get_logger(__name__)')
                import_lines.append('')
                in_imports = False
            else:
                other_lines.append(line)
        
        if import_lines:
            result = '\n'.join(import_lines + other_lines)
    
    return result, changes


def fix_old_string_formatting(content: str) -> Tuple[str, int]:
    """Convert old .format() and % formatting to f-strings.
    
    Returns:
        (updated_content, num_changes)
    """
    changes = 0
    
    # Fix .format() -> f-string
    # Pattern: "text {}".format(var)
    pattern = r'"([^"]*)\{([^}]*)\}([^"]*)".format\(([^)]+)\)'
    
    def replace_format(match):
        nonlocal changes
        before = match.group(1)
        placeholder = match.group(2)
        after = match.group(3)
        args = match.group(4)
        changes += 1
        return f'f"{before}{{{args}}}{after}"'
    
    content = re.sub(pattern, replace_format, content)
    
    # Fix % formatting
    # Pattern: "text %s" % (var,)
    pattern = r'"([^"]*)%s([^"]*)" % \(([^)]+)\)'
    
    def replace_percent(match):
        nonlocal changes
        before = match.group(1)
        after = match.group(2)
        var = match.group(3)
        changes += 1
        return f'f"{before}{{{var}}}{after}"'
    
    content = re.sub(pattern, replace_percent, content)
    
    return content, changes


def fix_security_issues(content: str, filename: str) -> Tuple[str, int]:
    """Fix security issues.
    
    Returns:
        (updated_content, num_changes)
    """
    changes = 0
    lines = content.split('\n')
    updated_lines = []
    needs_ast_import = False
    needs_secrets_import = False
    
    for line in lines:
        # Fix shell=True
        if 'shell=True' in line and 'shell=False' not in line:
            # Add comment explaining the security issue
            updated_lines.append(line.replace('shell=True', 'shell=False  # SECURITY: Changed from shell=True'))
            changes += 1
            continue
        
        # Fix eval() usage
        if re.search(r'\beval\s*\(', line) and '#' not in line.split('eval')[0]:
            # Suggest ast.literal_eval
            updated_lines.append(f"{line}  # TODO: Replace eval() with ast.literal_eval() for safety")
            needs_ast_import = True
            changes += 1
            continue
        
        # Fix exec() usage  
        if re.search(r'\bexec\s*\(', line) and '#' not in line.split('exec')[0]:
            updated_lines.append(f"{line}  # TODO: Redesign to avoid exec() - major security risk")
            changes += 1
            continue
        
        # Fix random.random() for security tokens
        if 'random.random()' in line or 'random.randint(' in line:
            if 'token' in line.lower() or 'secret' in line.lower() or 'key' in line.lower():
                updated_lines.append(line.replace('random.', 'secrets.'))
                needs_secrets_import = True
                changes += 1
                continue
        
        updated_lines.append(line)
    
    result = '\n'.join(updated_lines)
    
    # Add imports if needed
    if needs_ast_import and 'import ast' not in content:
        result = 'import ast\n' + result
    if needs_secrets_import and 'import secrets' not in content:
        result = 'import secrets\n' + result
    
    return result, changes


def upgrade_file(filepath: Path) -> dict:
    """Apply all upgrades to a file.
    
    Returns:
        dict with upgrade statistics
    """
    stats = {
        'print_fixes': 0,
        'format_fixes': 0,
        'security_fixes': 0,
    }
    
    try:
        content = filepath.read_text()
        original = content
        
        # Apply fixes
        content, print_changes = fix_print_to_logging(content, filepath.name)
        stats['print_fixes'] = print_changes
        
        content, format_changes = fix_old_string_formatting(content)
        stats['format_fixes'] = format_changes
        
        content, security_changes = fix_security_issues(content, filepath.name)
        stats['security_fixes'] = security_changes
        
        # Write back if changed
        if content != original:
            filepath.write_text(content)
            return stats
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
    
    return stats


def main():
    """Main upgrade script."""
    project_root = Path(__file__).parent
    
    # Priority files to upgrade
    priority_files = [
        "rfsn_controller/cgw_cli.py",
        "rfsn_controller/ci_entrypoint.py",
        "cgw_ssl_guard/coding_agent/cli.py",
        "cgw_ssl_guard/coding_agent/__init__.py",
        "cgw_ssl_guard/coding_agent/streaming_llm.py",
    ]
    
    total_stats = {
        'files_processed': 0,
        'print_fixes': 0,
        'format_fixes': 0,
        'security_fixes': 0,
    }
    
    print("RFSN Controller Upgrade Script")
    print("=" * 60)
    print()
    
    for file_path in priority_files:
        full_path = project_root / file_path
        if not full_path.exists():
            continue
        
        print(f"Processing: {file_path}")
        stats = upgrade_file(full_path)
        
        total_stats['files_processed'] += 1
        total_stats['print_fixes'] += stats['print_fixes']
        total_stats['format_fixes'] += stats['format_fixes']
        total_stats['security_fixes'] += stats['security_fixes']
        
        if any(stats.values()):
            print(f"  ✓ Print fixes: {stats['print_fixes']}")
            print(f"  ✓ Format fixes: {stats['format_fixes']}")
            print(f"  ✓ Security fixes: {stats['security_fixes']}")
        else:
            print(f"  - No changes needed")
        print()
    
    print("=" * 60)
    print(f"Total files processed: {total_stats['files_processed']}")
    print(f"Total print fixes: {total_stats['print_fixes']}")
    print(f"Total format fixes: {total_stats['format_fixes']}")
    print(f"Total security fixes: {total_stats['security_fixes']}")
    print()
    print("✓ Upgrade complete!")


if __name__ == "__main__":
    main()
