import streamlit as st
import geemap.foliumap as geemap
import ee
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import os
import zipfile
import tempfile
import shapely.geometry
from datetime import datetime
import json

# --- CONFIG & STYLING ---
st.set_page_config(
    page_title="GEE Vegetation Index Monitor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Global Typography */
    html, body, [data-testid="stSidebar"], .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Title Styling */
    .title-gradient {
        background: linear-gradient(135deg, #1a9641 0%, #52b788 50%, #2171b5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        color: #888888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }
    
    /* Card Glassmorphism Style */
    .glass-card {
        background: rgba(22, 27, 34, 0.6);
        backdrop-filter: blur(8px);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Custom Metric Styling */
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #52b788;
        margin-bottom: 0px;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Map Container Style */
    iframe {
        border-radius: 12px;
        border: 1px solid #30363d !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- GOOGLE EARTH ENGINE INITIALIZATION ---
@st.cache_resource
def init_gee():
    # 1. Try Service Account from secrets
    if 'EE_SERVICE_ACCOUNT' in st.secrets:
        try:
            sa_info = json.loads(st.secrets['EE_SERVICE_ACCOUNT'])
            credentials = ee.ServiceAccountCredentials(
                sa_info['client_email'],
                key_data=sa_info['private_key']
            )
            ee.Initialize(credentials)
            return True
        except Exception as e:
            st.error(f"Gagal inisialisasi GEE via Service Account: {e}")
            
    # 2. Try Default Credentials / ee.Authenticate() fallback
    try:
        ee.Initialize()
        return True
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize()
            return True
        except Exception as e:
            st.error(f"Otentikasi GEE gagal. Harap konfigurasikan Service Account atau jalankan ee.Authenticate() secara manual. Error: {e}")
            return False

# --- AOI FILE PARSER ---
def parse_aoi(uploaded_file):
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if ext == '.zip':
            # Extract zip containing shapefile
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            shp_files = [f for f in os.listdir(tmpdir) if f.endswith('.shp')]
            if not shp_files:
                raise ValueError("Tidak ditemukan berkas .shp di dalam file ZIP yang diunggah.")
            read_path = os.path.join(tmpdir, shp_files[0])
        else:
            read_path = filepath
            
        try:
            # Try Pyogrio first as it handles KML nicely without explicit driver enabling
            gdf = gpd.read_file(read_path, engine="pyogrio")
        except Exception as e:
            try:
                # Fallback to fiona, explicitly enabling KML
                import fiona
                fiona.drvsupport.supported_drivers['KML'] = 'rw'
                fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
                gdf = gpd.read_file(read_path)
            except Exception as e2:
                raise ValueError(f"Gagal membaca format spasial: Pyogrio ({e}) | Fiona ({e2})")
            
        if gdf.empty:
            raise ValueError("Berkas spasial yang diunggah tidak memiliki data geometri.")
            
        # Automatic WGS84 Reprojection
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
            
        # Fix invalid geometries and keep only Polygons (GEE reducers need area)
        gdf.geometry = gdf.geometry.make_valid()
        gdf = gdf.explode(index_parts=True)
        gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if gdf.empty:
            raise ValueError("Berkas AOI tidak memiliki tipe geometri area/Polygon yang valid.")
            
        return gdf

# --- GEE IMAGERY PROCESSING ---
def mask_s2_clouds(image, cloud_threshold):
    # Cloud mask based on QA60 band (Bit 10: Clouds, Bit 11: Cirrus)
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    qa_mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    
    # Cloud probability mask (from COPERNICUS/S2_CLOUD_PROBABILITY)
    cloud_prob = ee.Image(image.get('cloud_probability')).select('probability')
    prob_mask = cloud_prob.lte(cloud_threshold)
    
    return image.updateMask(qa_mask).updateMask(prob_mask)

def get_s2_collection(aoi, start_date, end_date, cloud_threshold):
    s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
              .filterBounds(aoi) \
              .filterDate(start_date, end_date)
              
    s2_clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
                  .filterBounds(aoi) \
                  .filterDate(start_date, end_date)
                  
    # Join collections on system:index
    join_filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
    join = ee.Join.saveFirst('cloud_probability')
    joined_col = join.apply(s2_sr, s2_clouds, join_filter)
    
    return ee.ImageCollection(joined_col).map(lambda img: mask_s2_clouds(img, cloud_threshold))

def add_index(image, index_name):
    if index_name == 'NDVI':
        # (B8 - B4) / (B8 + B4)
        index = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    elif index_name == 'EVI':
        # 2.5 * ((B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0))
        # Reflectance needs to be normalized to 0-1 range (divide by 10000)
        nir = image.select('B8').divide(10000.0)
        red = image.select('B4').divide(10000.0)
        blue = image.select('B2').divide(10000.0)
        index = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))',
            {
                'NIR': nir,
                'RED': red,
                'BLUE': blue
            }
        ).rename('EVI')
    elif index_name == 'NDWI':
        # McFeeters: (B3 - B8) / (B3 + B8) (Water Index)
        index = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    else:
        raise ValueError(f"Index {index_name} is not supported.")
        
    return image.addBands(index)

# --- CACHED TIMESERIES EXTRACTION ---
# We cache this to avoid requesting data from GEE repeatedly for the same parameters.
@st.cache_data(ttl=86400) # 24 hours TTL
def fetch_gee_data(geojson_geom, cloud_threshold, index_name, start_date, end_date):
    if not init_gee():
        return []
        
    aoi = ee.Geometry(geojson_geom)
    collection = get_s2_collection(aoi, start_date, end_date, cloud_threshold)
    collection_with_index = collection.map(lambda img: add_index(img, index_name))
    
    def calculate_spatial_mean(img):
        mean_val = img.select(index_name).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=30, # 30m scale for responsive computation speed
            maxPixels=1e9
        ).get(index_name)
        
        return ee.Feature(None, {
            'date': img.date().format('yyyy-MM-dd'),
            'mean': mean_val
        })
        
    stats_features = collection_with_index.map(calculate_spatial_mean).filter(ee.Filter.notNull(['mean']))
    
    try:
        info = stats_features.getInfo()
    except Exception as e:
        # Fallback or propagate error
        raise RuntimeError(f"Gagal mengambil data dari Google Earth Engine: {e}")
        
    data = []
    for feat in info.get('features', []):
        props = feat['properties']
        data.append({
            'date': props['date'],
            'mean': props['mean']
        })
        
    return sorted(data, key=lambda x: x['date'])

# --- APP LAYOUT ---
def main():
    st.markdown('<div class="title-gradient">🌿 Vegetation Index Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle-text">Pemantauan Histori Indeks Vegetasi Dinamis berbasis Google Earth Engine & Sentinel-2</div>', unsafe_allow_html=True)
    
    # Initialize GEE first
    gee_status = init_gee()
    if not gee_status:
        st.warning("⚠️ Google Earth Engine belum diinisialisasi. Silakan lengkapi otentikasi.")
        return
        
    # --- SIDEBAR CONTROL PANEL ---
    st.sidebar.markdown("### ⚙️ Panel Kontrol")
    
    # 1. AOI Upload
    st.sidebar.markdown("---")
    st.sidebar.markdown("**1. Batas Wilayah (AOI)**")
    uploaded_file = st.sidebar.file_uploader(
        "Unggah file AOI (.geojson, .kml, atau .zip berisi Shapefile)",
        type=["geojson", "kml", "zip"]
    )
    
    # 2. Select Index
    st.sidebar.markdown("---")
    st.sidebar.markdown("**2. Indeks Spektroskopi**")
    index_option = st.sidebar.selectbox(
        "Pilih Indeks:",
        options=["NDVI", "EVI", "NDWI"],
        format_func=lambda x: {
            "NDVI": "NDVI (Kesehatan Vegetasi)",
            "EVI": "EVI (Kerapatan Kanopi)",
            "NDWI": "NDWI (Indeks Air)"
        }[x]
    )
    
    # Descriptions of indices
    index_desc = {
        "NDVI": "Normalized Difference Vegetation Index digunakan untuk mengukur kehijauan dan kesehatan tanaman.",
        "EVI": "Enhanced Vegetation Index mengoreksi distorsi atmosfer dan pengaruh tanah, sangat baik untuk area kanopi padat.",
        "NDWI": "Normalized Difference Water Index sensitif terhadap kandungan air tanaman dan genangan air permukaan."
    }
    st.sidebar.info(index_desc[index_option])
    
    # 3. Cloud Cover Threshold
    st.sidebar.markdown("---")
    st.sidebar.markdown("**3. Toleransi Tutupan Awan**")
    cloud_threshold = st.sidebar.slider(
        "Maksimum Probabilitas Awan (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=5
    )
    
    # 4. Map Layer View Option
    st.sidebar.markdown("---")
    st.sidebar.markdown("**4. Tampilan Layer Peta**")
    map_layer_type = st.sidebar.radio(
        "Pilih Layer:",
        options=["True Color (RGB B4/B3/B2)", "Spectral Index Visualization"]
    )
    
    # --- DYNAMIC DATES AND TIMELINE ---
    # Start date is hardcoded to '2025-07-01' as requested
    start_date = "2025-07-01"
    # End date is dynamically set to today
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            gdf = parse_aoi(uploaded_file)
            # Create ee.Geometry from gdf
            union_geom = gdf.geometry.unary_union
            
            # Sederhanakan geometri untuk GeoJSON/KML yang sangat kompleks agar tidak melebihi batas payload GEE
            # Toleransi 0.0001 derajat (~10 meter) cukup untuk mempertahankan bentuk asli namun memangkas jumlah vertex
            union_geom = union_geom.simplify(tolerance=0.0001, preserve_topology=True)
            
            geojson_geom = shapely.geometry.mapping(union_geom)
            
            # Show success info
            st.success(f"✓ Berhasil memuat {uploaded_file.name}. Memproses data GEE...")
            
            # Fetch Time Series Data (Cached)
            with st.spinner("Mengunduh data Sentinel-2 dari GEE..."):
                try:
                    timeseries_data = fetch_gee_data(
                        geojson_geom=geojson_geom,
                        cloud_threshold=cloud_threshold,
                        index_name=index_option,
                        start_date=start_date,
                        end_date=end_date
                    )
                except Exception as ex:
                    st.error(f"Gagal memuat citra GEE: {ex}")
                    return
            
            if not timeseries_data:
                st.warning("⚠️ Tidak ada citra yang bebas awan pada rentang waktu terpilih untuk area ini. Coba naikkan toleransi tutupan awan di sidebar.")
                return
                
            df = pd.DataFrame(timeseries_data)
            dates = df['date'].tolist()
            
            # Setup columns for Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Jumlah Citra</div>
                    <div class="metric-value">{len(df)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Rata-rata {index_option}</div>
                    <div class="metric-value">{df['mean'].mean():.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Maksimum {index_option}</div>
                    <div class="metric-value">{df['mean'].max():.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_m4:
                st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Minimum {index_option}</div>
                    <div class="metric-value">{df['mean'].min():.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # --- MAP VIEW ---
            st.markdown("### 🗺️ Visualisasi Peta Citra")
            
            # Timeline slider located directly below the map
            # Container for map and slider
            map_placeholder = st.empty()
            
            # Timeline Slider using st.select_slider
            selected_date = st.select_slider(
                "📅 Geser Timeline Tanggal Akuisisi Citra:",
                options=dates,
                value=dates[-1], # default to latest image
                key="timeline_date"
            )
            
            # Query GEE image for selected date
            ee_geom = ee.Geometry(geojson_geom)
            selected_date_start = ee.Date(selected_date)
            selected_date_end = selected_date_start.advance(1, 'day')
            
            # Fetch the selected image
            img_col = get_s2_collection(ee_geom, selected_date_start, selected_date_end, cloud_threshold)
            img_col_with_idx = img_col.map(lambda img: add_index(img, index_option))
            image = img_col_with_idx.first()
            
            # Render interactive map using geemap
            m = geemap.Map()
            m.centerObject(ee_geom, zoom=13)
            
            # Display appropriate layer
            if map_layer_type == "True Color (RGB B4/B3/B2)":
                # Scale from 0-3000 to standard reflectance
                vis_params = {
                    'bands': ['B4', 'B3', 'B2'],
                    'min': 0,
                    'max': 3000,
                    'gamma': 1.4
                }
                m.addLayer(image, vis_params, f"Sentinel-2 True Color ({selected_date})")
            else:
                # Spectral Index Layer
                if index_option == 'NDVI':
                    vis_params = {
                        'bands': ['NDVI'],
                        'min': -0.1,
                        'max': 0.8,
                        'palette': ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
                    }
                elif index_option == 'EVI':
                    vis_params = {
                        'bands': ['EVI'],
                        'min': -0.1,
                        'max': 0.8,
                        'palette': ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']
                    }
                else: # NDWI
                    vis_params = {
                        'bands': ['NDWI'],
                        'min': -0.5,
                        'max': 0.5,
                        'palette': ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594']
                    }
                m.addLayer(image, vis_params, f"Sentinel-2 {index_option} ({selected_date})")
                
            # Add AOI border outline
            aoi_style = {'color': '#ff4b4b', 'fillColor': '00000000', 'weight': 2.5}
            m.addLayer(ee_geom, aoi_style, "Batas Wilayah (AOI)")
            
            # Draw Map
            with map_placeholder.container():
                m.to_streamlit(height=450)
                
            # --- PLOTLY TREND CHART ---
            st.markdown("---")
            st.markdown("### 📈 Grafik Tren Temporal Spasial")
            
            # Get selected index value for vline indicator
            selected_row = df[df['date'] == selected_date]
            selected_val = selected_row['mean'].values[0] if not selected_row.empty else None
            
            # Create Plotly interactive line chart
            fig = go.Figure()
            
            # Trend Line
            fig.add_trace(go.Scatter(
                x=df['date'],
                y=df['mean'],
                mode='lines+markers',
                name=f'Rata-rata Spasial {index_option}',
                line=dict(color='#1a9641', width=3),
                marker=dict(size=8, color='#52b788'),
                hovertemplate="Tanggal: %{x}<br>Nilai: %{y:.4f}<extra></extra>"
            ))
            
            # Add dynamic Vertical Line matching selected timeline date
            fig.add_vline(
                x=selected_date,
                line_width=2,
                line_dash="dash",
                line_color="#ff4b4b",
                annotation_text=f"Citra Terpilih ({selected_date})",
                annotation_position="top left",
                annotation_font=dict(color="#ff4b4b", size=11)
            )
            
            # Highlight selected point with a distinct marker
            if selected_val is not None:
                fig.add_trace(go.Scatter(
                    x=[selected_date],
                    y=[selected_val],
                    mode='markers',
                    name='Citra Terpilih',
                    marker=dict(size=12, color='#ff4b4b', symbol='star', line=dict(color='white', width=1)),
                    hoverinfo='skip'
                ))
                
            fig.update_layout(
                title=dict(
                    text=f"Distribusi Nilai Rata-Rata {index_option} di Area Terpilih",
                    font=dict(size=16, family="Outfit")
                ),
                xaxis=dict(
                    title="Tanggal Pengambilan Citra",
                    gridcolor="#30363d",
                    showgrid=True
                ),
                yaxis=dict(
                    title=f"Nilai Rata-Rata {index_option}",
                    gridcolor="#30363d",
                    showgrid=True
                ),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
                margin=dict(l=40, r=40, t=60, b=40),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")
            
    else:
        # Default placeholder when no file is uploaded
        st.info("👋 Selamat Datang! Silakan unggah berkas spasial batas wilayah (AOI) Anda pada panel kiri untuk memulai analisis.")
        
        # Guide cards
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.markdown("""
            <div class="glass-card">
                <h3>📂 Unggah Berkas</h3>
                <p>Mendukung format GeoJSON (.geojson), KML (.kml), atau Shapefile (.shp diunggah dalam format ZIP bersama file pendukungnya).</p>
            </div>
            """, unsafe_allow_html=True)
        with col_g2:
            st.markdown("""
            <div class="glass-card">
                <h3>☁️ Filter Awan Dinamis</h3>
                <p>Mengintegrasikan dataset <b>S2_CLOUD_PROBABILITY</b> dan QA60 untuk memastikan analisis Anda bebas dari tutupan awan pengganggu.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_g3:
            st.markdown("""
            <div class="glass-card">
                <h3>📊 Grafik Tren Terhubung</h3>
                <p>Menampilkan grafik pergerakan nilai spasial secara bulanan dengan garis indikator dinamis yang sinkron dengan peta.</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
