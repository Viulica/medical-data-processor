#!/usr/bin/env python3
"""
Debug script to explore SharePoint structure and find the correct path.
"""

import requests
import json

# Azure AD Application Credentials
CLIENT_ID = "a54eaddf-654d-4f0e-9071-2b8c8ad26942"
TENANT_ID = "0138a897-ff88-4dc2-933b-fad359609873"
CLIENT_SECRET = "MlO8Q~6ARQrwWgbvb6v9qWOuHCWIU3m6MaKsTczK"

# SharePoint details
SHAREPOINT_SITE = "anesthesiapartners.sharepoint.com"
SITE_PATH = "/sites/CodedWork9"


def get_access_token():
    """Obtain an access token from Azure AD."""
    token_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token"
    
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scope': 'https://graph.microsoft.com/.default'
    }
    
    response = requests.post(token_url, data=token_data)
    response.raise_for_status()
    return response.json()['access_token']


def get_site_id(access_token):
    """Get the SharePoint site ID."""
    site_url = f"https://graph.microsoft.com/v1.0/sites/{SHAREPOINT_SITE}:{SITE_PATH}"
    headers = {'Authorization': f'Bearer {access_token}'}
    
    response = requests.get(site_url, headers=headers)
    response.raise_for_status()
    return response.json()['id']


def list_drives(access_token, site_id):
    """List all drives in the site."""
    drives_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives"
    headers = {'Authorization': f'Bearer {access_token}'}
    
    response = requests.get(drives_url, headers=headers)
    response.raise_for_status()
    return response.json()


def list_drive_root(access_token, site_id, drive_id):
    """List root contents of a specific drive."""
    root_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drives/{drive_id}/root/children"
    headers = {'Authorization': f'Bearer {access_token}'}
    
    response = requests.get(root_url, headers=headers)
    response.raise_for_status()
    return response.json()


def try_folder_path(access_token, site_id, drive_id, path):
    """Try to access a specific folder path."""
    # Try different path formats
    paths_to_try = [
        f"/drive/root:{path}:/children",
        f"/drives/{drive_id}/root:{path}:/children",
        f"/drive/root:/{path}:/children",
        f"/drives/{drive_id}/root:/{path}:/children",
    ]
    
    for test_path in paths_to_try:
        url = f"https://graph.microsoft.com/v1.0/sites/{site_id}{test_path}"
        headers = {'Authorization': f'Bearer {access_token}'}
        
        print(f"\nTrying: {url}")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            print(f"✅ SUCCESS with path: {test_path}")
            return response.json()
        else:
            print(f"❌ Failed: {response.status_code}")
    
    return None


def main():
    print("SharePoint Structure Debug Script")
    print("=" * 80)
    
    # Get access token
    print("\n1. Getting access token...")
    access_token = get_access_token()
    print("✅ Token obtained")
    
    # Get site ID
    print("\n2. Getting site ID...")
    site_id = get_site_id(access_token)
    print(f"✅ Site ID: {site_id}")
    
    # List all drives
    print("\n3. Listing all drives in the site...")
    drives = list_drives(access_token, site_id)
    
    print(f"\nFound {len(drives.get('value', []))} drive(s):")
    for drive in drives.get('value', []):
        print(f"\n  📁 Drive: {drive.get('name')}")
        print(f"     ID: {drive.get('id')}")
        print(f"     Type: {drive.get('driveType')}")
        print(f"     Web URL: {drive.get('webUrl')}")
        
        # List root contents of this drive
        drive_id = drive.get('id')
        print(f"\n     Root contents:")
        try:
            root_contents = list_drive_root(access_token, site_id, drive_id)
            for item in root_contents.get('value', [])[:5]:  # Show first 5 items
                item_type = "📁" if 'folder' in item else "📄"
                print(f"       {item_type} {item.get('name')}")
        except Exception as e:
            print(f"       ❌ Error: {e}")
    
    # Try to access the specific folder
    print("\n" + "=" * 80)
    print("4. Trying to access folder: IAS/2025/MOR")
    print("=" * 80)
    
    if drives.get('value'):
        main_drive = drives['value'][0]  # Usually the first drive is "Documents"
        drive_id = main_drive.get('id')
        
        # Try different path variations
        paths = [
            "IAS/2025/MOR",
            "/IAS/2025/MOR",
            "Shared Documents/IAS/2025/MOR",
            "/Shared Documents/IAS/2025/MOR",
        ]
        
        for path in paths:
            print(f"\n--- Testing path: {path} ---")
            result = try_folder_path(access_token, site_id, drive_id, path)
            if result:
                print(f"\n✅ FOUND IT! Contents:")
                for item in result.get('value', [])[:10]:
                    item_type = "📁" if 'folder' in item else "📄"
                    print(f"  {item_type} {item.get('name')}")
                break


if __name__ == "__main__":
    main()
