#!/usr/bin/env python3
"""
Script to list folder contents from SharePoint using Azure AD authentication.
"""

import requests
from urllib.parse import quote
import json
import os

# Azure AD Application Credentials
CLIENT_ID = "a54eaddf-654d-4f0e-9071-2b8c8ad26942"
OBJECT_ID = "58b6d500-72e5-46a2-9020-be4ff8c31be3"
TENANT_ID = "0138a897-ff88-4dc2-933b-fad359609873"

# SharePoint details
SHAREPOINT_SITE = "anesthesiapartners.sharepoint.com"
SITE_PATH = "/sites/CodedWork9"
# Note: FOLDER_PATH should be relative to the document library root
# The "Shared Documents" part is implicit in the drive
FOLDER_PATH = "/IAS/2025/MOR"

# Client secret from Azure Portal (Certificates & secrets)
CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
if not CLIENT_SECRET:
    raise ValueError("AZURE_CLIENT_SECRET environment variable is not set")


def get_access_token():
    """
    Obtain an access token from Azure AD using client credentials flow.
    """
    token_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scope': 'https://graph.microsoft.com/.default'
    }
    
    try:
        response = requests.post(token_url, data=token_data)
        response.raise_for_status()
        token_response = response.json()
        return token_response['access_token']
    except requests.exceptions.RequestException as e:
        print(f"Error obtaining access token: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None


def get_site_id(access_token):
    """
    Get the SharePoint site ID using Microsoft Graph API.
    """
    # Use the hostname and site path to construct the Graph API URL
    site_url = f"https://graph.microsoft.com/v1.0/sites/{SHAREPOINT_SITE}:{SITE_PATH}"
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(site_url, headers=headers)
        response.raise_for_status()
        site_data = response.json()
        return site_data['id']
    except requests.exceptions.RequestException as e:
        print(f"Error getting site ID: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None


def list_folder_contents(access_token, site_id):
    """
    List contents of the specified SharePoint folder using Microsoft Graph API.
    """
    # Using Microsoft Graph API to list drive items
    folder_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive/root:{FOLDER_PATH}:/children"
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/json'
    }
    
    try:
        response = requests.get(folder_url, headers=headers)
        response.raise_for_status()
        items = response.json()
        return items
    except requests.exceptions.RequestException as e:
        print(f"Error listing folder contents: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None


def display_items(items):
    """
    Display the folder contents in a readable format.
    """
    if not items or 'value' not in items:
        print("No items found or error occurred.")
        return
    
    print(f"\n{'='*80}")
    print(f"Folder Contents: {FOLDER_PATH}")
    print(f"{'='*80}\n")
    
    folders = []
    files = []
    
    for item in items['value']:
        if 'folder' in item:
            folders.append(item)
        else:
            files.append(item)
    
    # Display folders
    if folders:
        print(f"📁 FOLDERS ({len(folders)}):")
        print(f"{'-'*80}")
        for folder in folders:
            name = folder.get('name', 'Unknown')
            created = folder.get('createdDateTime', 'Unknown')
            modified = folder.get('lastModifiedDateTime', 'Unknown')
            print(f"  📁 {name}")
            print(f"     Created: {created}")
            print(f"     Modified: {modified}")
            print()
    
    # Display files
    if files:
        print(f"\n📄 FILES ({len(files)}):")
        print(f"{'-'*80}")
        for file in files:
            name = file.get('name', 'Unknown')
            size = file.get('size', 0)
            created = file.get('createdDateTime', 'Unknown')
            modified = file.get('lastModifiedDateTime', 'Unknown')
            web_url = file.get('webUrl', 'No URL available')
            
            # Convert size to human-readable format
            size_str = format_size(size)
            
            print(f"  📄 {name}")
            print(f"     Size: {size_str}")
            print(f"     Created: {created}")
            print(f"     Modified: {modified}")
            print(f"     Link: {web_url}")
            print()
    
    print(f"{'='*80}")
    print(f"Total: {len(folders)} folder(s), {len(files)} file(s)")
    print(f"{'='*80}\n")


def format_size(size_bytes):
    """
    Convert bytes to human-readable format.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def save_to_json(items, filename='sharepoint_folder_contents.json'):
    """
    Save the folder contents to a JSON file.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        print(f"✅ Folder contents saved to: {filename}")
    except Exception as e:
        print(f"❌ Error saving to JSON: {e}")


def save_links_to_csv(items, filename='sharepoint_file_links.csv'):
    """
    Save file names and download links to a CSV file.
    """
    import csv
    
    try:
        files = [item for item in items.get('value', []) if 'folder' not in item]
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['File Name', 'Size (Bytes)', 'Size (Human)', 'Created', 'Modified', 'Download Link'])
            
            for file in files:
                name = file.get('name', 'Unknown')
                size = file.get('size', 0)
                size_str = format_size(size)
                created = file.get('createdDateTime', 'Unknown')
                modified = file.get('lastModifiedDateTime', 'Unknown')
                web_url = file.get('webUrl', 'No URL')
                
                writer.writerow([name, size, size_str, created, modified, web_url])
        
        print(f"✅ File links saved to: {filename}")
        print(f"   Total files: {len(files)}")
    except Exception as e:
        print(f"❌ Error saving to CSV: {e}")


def main():
    """
    Main function to orchestrate the SharePoint folder listing.
    """
    print("SharePoint Folder Listing Script")
    print("=" * 80)
    
    # Step 1: Get access token
    print("\n1. Obtaining access token...")
    access_token = get_access_token()
    if not access_token:
        print("❌ Failed to obtain access token. Exiting.")
        return
    print("✅ Access token obtained successfully.")
    
    # Step 2: Get site ID
    print("\n2. Getting SharePoint site ID...")
    site_id = get_site_id(access_token)
    if not site_id:
        print("❌ Failed to get site ID. Exiting.")
        return
    print(f"✅ Site ID: {site_id}")
    
    # Step 3: List folder contents
    print("\n3. Listing folder contents...")
    items = list_folder_contents(access_token, site_id)
    if not items:
        print("❌ Failed to list folder contents. Exiting.")
        return
    
    # Step 4: Display items
    display_items(items)
    
    # Step 5: Save options
    print("\n" + "="*80)
    print("Save Options:")
    print("="*80)
    
    # Save to CSV with links
    csv_choice = input("\nWould you like to save file links to a CSV file? (y/n): ")
    if csv_choice.lower() == 'y':
        save_links_to_csv(items)
    
    # Save to JSON (full data)
    json_choice = input("\nWould you like to save the full results to a JSON file? (y/n): ")
    if json_choice.lower() == 'y':
        save_to_json(items)
    
    print("\n✅ Script completed successfully!")


if __name__ == "__main__":
    main()
