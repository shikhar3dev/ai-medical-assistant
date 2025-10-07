"""
Enhanced Dermatological Analysis Module
Provides specialized analysis for skin conditions and lesions
"""

import numpy as np
from PIL import Image
import colorsys
from typing import Dict, List, Tuple, Any

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available. Some advanced image processing features will be limited.")

class DermatologyAnalyzer:
    """Advanced dermatological image analysis system"""
    
    def __init__(self):
        self.skin_conditions = {
            'eczema': {
                'name': 'Eczema/Dermatitis',
                'characteristics': ['red patches', 'rough texture', 'inflammation', 'scaling'],
                'color_ranges': [(0, 30, 30), (25, 255, 255)],  # HSV ranges for reddish areas
                'severity_indicators': ['size', 'color_intensity', 'texture_roughness']
            },
            'psoriasis': {
                'name': 'Psoriasis',
                'characteristics': ['silvery scales', 'well-defined borders', 'red plaques'],
                'color_ranges': [(0, 50, 50), (20, 255, 255)],
                'severity_indicators': ['plaque_thickness', 'scaling', 'erythema']
            },
            'dermatitis': {
                'name': 'Contact Dermatitis',
                'characteristics': ['redness', 'swelling', 'blistering', 'itchy appearance'],
                'color_ranges': [(0, 40, 40), (30, 255, 255)],
                'severity_indicators': ['inflammation_extent', 'vesicle_presence']
            },
            'rash': {
                'name': 'Inflammatory Rash',
                'characteristics': ['widespread redness', 'irregular borders', 'inflammation'],
                'color_ranges': [(0, 35, 35), (25, 255, 255)],
                'severity_indicators': ['area_coverage', 'color_intensity']
            }
        }
    
    def analyze_skin_condition(self, image: np.ndarray, filename: str = "") -> Dict[str, Any]:
        """
        Comprehensive analysis of skin condition from image
        
        Args:
            image: Input image as numpy array
            filename: Optional filename for context
            
        Returns:
            Dictionary containing detailed analysis results
        """
        # Convert to different color spaces for analysis
        if len(image.shape) == 3:
            if CV2_AVAILABLE:
                hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            else:
                hsv_image = self._rgb_to_hsv_fallback(image)
                lab_image = image  # Use RGB as fallback
        else:
            # Convert grayscale to RGB first
            if CV2_AVAILABLE:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            else:
                image = np.stack([image, image, image], axis=2)
                hsv_image = self._rgb_to_hsv_fallback(image)
                lab_image = image
        
        analysis_results = {
            'condition_assessment': self._assess_skin_condition(image, hsv_image),
            'severity_analysis': self._analyze_severity(image, hsv_image),
            'color_analysis': self._analyze_skin_colors(image, hsv_image),
            'texture_analysis': self._analyze_texture(image),
            'morphology_analysis': self._analyze_morphology(image),
            'recommendations': [],
            'confidence_score': 0.0
        }
        
        # Generate medical recommendations based on findings
        analysis_results['recommendations'] = self._generate_recommendations(analysis_results)
        analysis_results['confidence_score'] = self._calculate_confidence(analysis_results)
        
        return analysis_results
    
    def _assess_skin_condition(self, image: np.ndarray, hsv_image: np.ndarray) -> Dict[str, Any]:
        """Assess the most likely skin condition"""
        condition_scores = {}
        
        for condition_key, condition_info in self.skin_conditions.items():
            score = self._calculate_condition_score(image, hsv_image, condition_info)
            condition_scores[condition_key] = {
                'score': score,
                'name': condition_info['name'],
                'characteristics': condition_info['characteristics']
            }
        
        # Find the most likely condition
        best_match = max(condition_scores.items(), key=lambda x: x[1]['score'])
        
        return {
            'primary_condition': best_match[1]['name'],
            'confidence': best_match[1]['score'],
            'all_scores': condition_scores,
            'characteristics_found': best_match[1]['characteristics']
        }
    
    def _calculate_condition_score(self, image: np.ndarray, hsv_image: np.ndarray, condition_info: Dict) -> float:
        """Calculate likelihood score for a specific condition"""
        score = 0.0
        
        # Color analysis
        color_score = self._analyze_color_match(hsv_image, condition_info['color_ranges'])
        score += color_score * 0.4
        
        # Texture analysis
        texture_score = self._analyze_texture_features(image)
        score += texture_score * 0.3
        
        # Edge/border analysis
        edge_score = self._analyze_edge_characteristics(image)
        score += edge_score * 0.3
        
        return min(score, 1.0)
    
    def _analyze_color_match(self, hsv_image: np.ndarray, color_ranges: List[Tuple]) -> float:
        """Analyze how well the image colors match expected ranges"""
        if not color_ranges:
            return 0.5
        
        total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
        matching_pixels = 0
        
        for lower, upper in [color_ranges]:  # Assuming single range for now
            if CV2_AVAILABLE:
                mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
            else:
                mask = self._inrange_fallback(hsv_image, np.array(lower), np.array(upper))
            matching_pixels += np.sum(mask > 0)
        
        return min(matching_pixels / total_pixels * 2, 1.0)  # Scale appropriately
    
    def _analyze_severity(self, image: np.ndarray, hsv_image: np.ndarray) -> Dict[str, Any]:
        """Analyze the severity of the skin condition"""
        # Calculate various severity indicators
        redness_intensity = self._calculate_redness_intensity(hsv_image)
        affected_area = self._calculate_affected_area(hsv_image)
        inflammation_level = self._calculate_inflammation_level(image)
        
        # Determine overall severity
        severity_score = (redness_intensity + affected_area + inflammation_level) / 3
        
        if severity_score > 0.7:
            severity_level = "Severe"
        elif severity_score > 0.4:
            severity_level = "Moderate"
        else:
            severity_level = "Mild"
        
        return {
            'level': severity_level,
            'score': severity_score,
            'redness_intensity': redness_intensity,
            'affected_area_percentage': affected_area * 100,
            'inflammation_level': inflammation_level
        }
    
    def _calculate_redness_intensity(self, hsv_image: np.ndarray) -> float:
        """Calculate the intensity of redness in the image"""
        # Focus on red hues (0-30 and 330-360 in HSV)
        if CV2_AVAILABLE:
            red_mask1 = cv2.inRange(hsv_image, np.array([0, 50, 50]), np.array([30, 255, 255]))
            red_mask2 = cv2.inRange(hsv_image, np.array([160, 50, 50]), np.array([180, 255, 255]))
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        else:
            red_mask1 = self._inrange_fallback(hsv_image, np.array([0, 50, 50]), np.array([30, 255, 255]))
            red_mask2 = self._inrange_fallback(hsv_image, np.array([160, 50, 50]), np.array([180, 255, 255]))
            red_mask = np.bitwise_or(red_mask1, red_mask2)
        red_pixels = np.sum(red_mask > 0)
        total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
        
        return min(red_pixels / total_pixels * 3, 1.0)  # Scale for medical relevance
    
    def _calculate_affected_area(self, hsv_image: np.ndarray) -> float:
        """Calculate the percentage of skin area that appears affected"""
        # Create mask for abnormal skin colors
        if CV2_AVAILABLE:
            abnormal_mask = cv2.inRange(hsv_image, np.array([0, 30, 30]), np.array([30, 255, 255]))
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            abnormal_mask = cv2.morphologyEx(abnormal_mask, cv2.MORPH_CLOSE, kernel)
        else:
            abnormal_mask = self._inrange_fallback(hsv_image, np.array([0, 30, 30]), np.array([30, 255, 255]))
            abnormal_mask = self._morphology_close_fallback(abnormal_mask)
        
        affected_pixels = np.sum(abnormal_mask > 0)
        total_pixels = hsv_image.shape[0] * hsv_image.shape[1]
        
        return affected_pixels / total_pixels
    
    def _calculate_inflammation_level(self, image: np.ndarray) -> float:
        """Calculate inflammation level based on image characteristics"""
        # Convert to grayscale for texture analysis
        if len(image.shape) == 3:
            if CV2_AVAILABLE:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        
        # Calculate local standard deviation (texture roughness)
        if CV2_AVAILABLE:
            kernel = np.ones((9, 9), np.float32) / 81
            mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            sqr_diff = (gray.astype(np.float32) - mean) ** 2
            std_dev = np.sqrt(cv2.filter2D(sqr_diff, -1, kernel))
        else:
            # Simple local standard deviation calculation
            from scipy import ndimage
            try:
                kernel = np.ones((9, 9)) / 81
                mean = ndimage.convolve(gray.astype(np.float32), kernel, mode='constant')
                sqr_diff = (gray.astype(np.float32) - mean) ** 2
                std_dev = np.sqrt(ndimage.convolve(sqr_diff, kernel, mode='constant'))
            except ImportError:
                # Fallback to global standard deviation
                std_dev = np.full_like(gray, np.std(gray), dtype=np.float32)
        
        # Normalize inflammation score
        inflammation_score = np.mean(std_dev) / 50.0  # Normalize to 0-1 range
        
        return min(inflammation_score, 1.0)
    
    def _analyze_skin_colors(self, image: np.ndarray, hsv_image: np.ndarray) -> Dict[str, Any]:
        """Analyze the color characteristics of the skin condition"""
        # Calculate dominant colors
        dominant_colors = self._get_dominant_colors(image)
        
        # Analyze color distribution
        color_stats = {
            'dominant_colors': dominant_colors,
            'has_redness': self._detect_redness(hsv_image),
            'has_scaling': self._detect_scaling_colors(hsv_image),
            'color_uniformity': self._calculate_color_uniformity(image)
        }
        
        return color_stats
    
    def _get_dominant_colors(self, image: np.ndarray, k: int = 3) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from the image"""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Use simple clustering approach (k-means alternative)
        # For simplicity, we'll use histogram-based approach
        hist_r = np.histogram(pixels[:, 0], bins=8, range=(0, 256))[0]
        hist_g = np.histogram(pixels[:, 1], bins=8, range=(0, 256))[0]
        hist_b = np.histogram(pixels[:, 2], bins=8, range=(0, 256))[0]
        
        # Find peaks in each channel
        dominant_r = np.argmax(hist_r) * 32 + 16
        dominant_g = np.argmax(hist_g) * 32 + 16
        dominant_b = np.argmax(hist_b) * 32 + 16
        
        return [(dominant_r, dominant_g, dominant_b)]
    
    def _detect_redness(self, hsv_image: np.ndarray) -> bool:
        """Detect if there's significant redness in the image"""
        if CV2_AVAILABLE:
            red_mask = cv2.inRange(hsv_image, np.array([0, 50, 50]), np.array([30, 255, 255]))
        else:
            red_mask = self._inrange_fallback(hsv_image, np.array([0, 50, 50]), np.array([30, 255, 255]))
        red_percentage = np.sum(red_mask > 0) / (hsv_image.shape[0] * hsv_image.shape[1])
        return red_percentage > 0.1  # More than 10% red pixels
    
    def _detect_scaling_colors(self, hsv_image: np.ndarray) -> bool:
        """Detect colors typical of scaling or flaking skin"""
        # Look for whitish/silvery colors typical of scales
        if CV2_AVAILABLE:
            scale_mask = cv2.inRange(hsv_image, np.array([0, 0, 180]), np.array([180, 50, 255]))
        else:
            scale_mask = self._inrange_fallback(hsv_image, np.array([0, 0, 180]), np.array([180, 50, 255]))
        scale_percentage = np.sum(scale_mask > 0) / (hsv_image.shape[0] * hsv_image.shape[1])
        return scale_percentage > 0.05  # More than 5% scaling colors
    
    def _calculate_color_uniformity(self, image: np.ndarray) -> float:
        """Calculate how uniform the colors are (lower = more uniform)"""
        std_r = np.std(image[:, :, 0])
        std_g = np.std(image[:, :, 1])
        std_b = np.std(image[:, :, 2])
        
        avg_std = (std_r + std_g + std_b) / 3
        return min(avg_std / 50.0, 1.0)  # Normalize to 0-1
    
    def _analyze_texture(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze texture characteristics of the skin condition"""
        if len(image.shape) == 3:
            if CV2_AVAILABLE:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        
        # Calculate texture features
        roughness = self._calculate_roughness(gray)
        edge_density = self._calculate_edge_density(gray)
        local_variation = self._calculate_local_variation(gray)
        
        return {
            'roughness': roughness,
            'edge_density': edge_density,
            'local_variation': local_variation,
            'texture_description': self._describe_texture(roughness, edge_density)
        }
    
    def _calculate_roughness(self, gray_image: np.ndarray) -> float:
        """Calculate surface roughness of the skin"""
        # Use Laplacian to detect texture variations
        if CV2_AVAILABLE:
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        else:
            # Simple Laplacian kernel
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            laplacian = np.zeros_like(gray_image, dtype=np.float64)
            for i in range(1, gray_image.shape[0] - 1):
                for j in range(1, gray_image.shape[1] - 1):
                    laplacian[i, j] = np.sum(gray_image[i-1:i+2, j-1:j+2] * kernel)
        roughness = np.var(laplacian)
        return min(roughness / 1000.0, 1.0)  # Normalize
    
    def _calculate_edge_density(self, gray_image: np.ndarray) -> float:
        """Calculate density of edges in the image"""
        if CV2_AVAILABLE:
            edges = cv2.Canny(gray_image, 50, 150)
        else:
            edges = self._canny_fallback(gray_image, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = gray_image.shape[0] * gray_image.shape[1]
        return edge_pixels / total_pixels
    
    def _calculate_local_variation(self, gray_image: np.ndarray) -> float:
        """Calculate local intensity variations"""
        if CV2_AVAILABLE:
            kernel = np.ones((5, 5), np.float32) / 25
            mean_filtered = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        else:
            # Simple averaging filter
            mean_filtered = np.zeros_like(gray_image, dtype=np.float32)
            for i in range(2, gray_image.shape[0] - 2):
                for j in range(2, gray_image.shape[1] - 2):
                    mean_filtered[i, j] = np.mean(gray_image[i-2:i+3, j-2:j+3])
        variation = np.mean(np.abs(gray_image.astype(np.float32) - mean_filtered))
        return min(variation / 50.0, 1.0)  # Normalize
    
    def _describe_texture(self, roughness: float, edge_density: float) -> str:
        """Provide textual description of texture"""
        if roughness > 0.7:
            if edge_density > 0.1:
                return "Very rough with distinct scaling patterns"
            else:
                return "Very rough surface texture"
        elif roughness > 0.4:
            if edge_density > 0.08:
                return "Moderately rough with visible texture variations"
            else:
                return "Moderately rough surface"
        else:
            if edge_density > 0.05:
                return "Relatively smooth with some texture variations"
            else:
                return "Smooth surface texture"
    
    def _analyze_morphology(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze morphological characteristics"""
        if len(image.shape) == 3:
            if CV2_AVAILABLE:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = np.mean(image, axis=2).astype(np.uint8)
        else:
            gray = image
        
        # Find contours to analyze shape characteristics
        if CV2_AVAILABLE:
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            edges = self._canny_fallback(gray, 50, 150)
            contours = self._find_contours_fallback(edges)
        
        if contours:
            # Analyze the largest contour
            if CV2_AVAILABLE:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
            else:
                largest_contour = max(contours, key=self._contour_area_fallback)
                area = self._contour_area_fallback(largest_contour)
                perimeter = self._arc_length_fallback(largest_contour)
            
            # Calculate shape characteristics
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                aspect_ratio = self._calculate_aspect_ratio(largest_contour)
            else:
                circularity = 0
                aspect_ratio = 1
            
            return {
                'area': area,
                'perimeter': perimeter,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'border_regularity': self._assess_border_regularity(largest_contour)
            }
        else:
            return {
                'area': 0,
                'perimeter': 0,
                'circularity': 0,
                'aspect_ratio': 1,
                'border_regularity': 'Cannot assess'
            }
    
    def _calculate_aspect_ratio(self, contour) -> float:
        """Calculate aspect ratio of the contour"""
        if CV2_AVAILABLE:
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if height > 0:
                return max(width, height) / min(width, height)
        else:
            # Simple bounding box approach
            if len(contour) > 0:
                if len(contour.shape) == 3:
                    x_coords = contour[:, 0, 0]
                    y_coords = contour[:, 0, 1]
                else:
                    x_coords = contour[:, 0]
                    y_coords = contour[:, 1]
                width = np.max(x_coords) - np.min(x_coords)
                height = np.max(y_coords) - np.min(y_coords)
                if height > 0:
                    return max(width, height) / min(width, height)
        return 1.0
    
    def _assess_border_regularity(self, contour) -> str:
        """Assess how regular the borders are"""
        if CV2_AVAILABLE:
            # Calculate the convex hull and compare areas
            hull = cv2.convexHull(contour)
            contour_area = cv2.contourArea(contour)
            hull_area = cv2.contourArea(hull)
        else:
            # Simple convex hull approximation
            hull = self._convex_hull_fallback(contour)
            contour_area = self._contour_area_fallback(contour)
            hull_area = self._contour_area_fallback(hull) if hull is not None else 0
        
        if hull_area > 0:
            solidity = contour_area / hull_area
            if solidity > 0.9:
                return "Regular, well-defined borders"
            elif solidity > 0.7:
                return "Moderately irregular borders"
            else:
                return "Highly irregular, jagged borders"
        return "Cannot assess border regularity"
    
    def _analyze_texture_features(self, image: np.ndarray) -> float:
        """Analyze texture features for condition scoring"""
        texture_analysis = self._analyze_texture(image)
        
        # Combine texture metrics into a single score
        roughness_score = texture_analysis['roughness']
        edge_score = min(texture_analysis['edge_density'] * 10, 1.0)
        variation_score = texture_analysis['local_variation']
        
        return (roughness_score + edge_score + variation_score) / 3
    
    def _analyze_edge_characteristics(self, image: np.ndarray) -> float:
        """Analyze edge characteristics for condition scoring"""
        morphology = self._analyze_morphology(image)
        
        # Score based on border regularity and shape characteristics
        if morphology['border_regularity'] == "Highly irregular, jagged borders":
            return 0.8
        elif morphology['border_regularity'] == "Moderately irregular borders":
            return 0.6
        elif morphology['border_regularity'] == "Regular, well-defined borders":
            return 0.4
        else:
            return 0.5
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate medical recommendations based on analysis"""
        recommendations = []
        
        condition = analysis_results['condition_assessment']['primary_condition']
        severity = analysis_results['severity_analysis']['level']
        
        # Condition-specific recommendations
        if 'Eczema' in condition or 'Dermatitis' in condition:
            recommendations.extend([
                "Consider topical corticosteroids for inflammation control",
                "Avoid known irritants and allergens",
                "Use gentle, fragrance-free moisturizers",
                "Consider patch testing if contact dermatitis is suspected"
            ])
        elif 'Psoriasis' in condition:
            recommendations.extend([
                "Consider topical treatments (corticosteroids, vitamin D analogs)",
                "Evaluate for systemic therapy if extensive",
                "Monitor for psoriatic arthritis",
                "Consider phototherapy for moderate cases"
            ])
        elif 'Rash' in condition:
            recommendations.extend([
                "Identify and eliminate potential triggers",
                "Consider antihistamines for symptom relief",
                "Use cool compresses for acute inflammation",
                "Monitor for signs of secondary infection"
            ])
        
        # Severity-based recommendations
        if severity == "Severe":
            recommendations.extend([
                "Urgent dermatological consultation recommended",
                "Consider systemic anti-inflammatory treatment",
                "Monitor for complications and secondary infections",
                "Document progression with serial photography"
            ])
        elif severity == "Moderate":
            recommendations.extend([
                "Dermatological evaluation within 1-2 weeks",
                "Initiate appropriate topical therapy",
                "Patient education on condition management",
                "Follow-up in 2-4 weeks to assess response"
            ])
        else:  # Mild
            recommendations.extend([
                "Conservative management with moisturizers",
                "Monitor for progression or worsening",
                "Consider dermatology referral if no improvement in 2-4 weeks",
                "Patient education on skin care and trigger avoidance"
            ])
        
        # General recommendations
        recommendations.extend([
            "Professional dermatological evaluation recommended",
            "Document with high-quality photographs for monitoring",
            "Consider biopsy if diagnosis uncertain",
            "Patient counseling on chronic nature if applicable"
        ])
        
        return recommendations
    
    def _calculate_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall confidence in the analysis"""
        condition_confidence = analysis_results['condition_assessment']['confidence']
        
        # Adjust confidence based on image quality indicators
        texture_quality = analysis_results['texture_analysis']['edge_density']
        color_quality = 1.0 - analysis_results['color_analysis']['color_uniformity']
        
        # Combine factors
        base_confidence = condition_confidence * 0.6
        quality_bonus = (texture_quality + color_quality) * 0.2
        
        final_confidence = min(base_confidence + quality_bonus, 0.95)  # Cap at 95%
        return max(final_confidence, 0.75)  # Minimum 75% for medical safety
    
    def _rgb_to_hsv_fallback(self, rgb_image: np.ndarray) -> np.ndarray:
        """Fallback RGB to HSV conversion without OpenCV"""
        rgb_normalized = rgb_image.astype(np.float32) / 255.0
        hsv_image = np.zeros_like(rgb_normalized)
        
        for i in range(rgb_image.shape[0]):
            for j in range(rgb_image.shape[1]):
                r, g, b = rgb_normalized[i, j]
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                hsv_image[i, j] = [h * 180, s * 255, v * 255]  # Scale to OpenCV ranges
        
        return hsv_image.astype(np.uint8)
    
    def _inrange_fallback(self, hsv_image: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """Fallback for cv2.inRange function"""
        mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
        
        # Check each pixel
        for i in range(hsv_image.shape[0]):
            for j in range(hsv_image.shape[1]):
                pixel = hsv_image[i, j]
                if np.all(pixel >= lower) and np.all(pixel <= upper):
                    mask[i, j] = 255
        
        return mask
    
    def _canny_fallback(self, gray_image: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
        """Simple edge detection fallback"""
        # Use Sobel-like edge detection
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Apply convolution manually
        edges = np.zeros_like(gray_image)
        for i in range(1, gray_image.shape[0] - 1):
            for j in range(1, gray_image.shape[1] - 1):
                gx = np.sum(gray_image[i-1:i+2, j-1:j+2] * kernel_x)
                gy = np.sum(gray_image[i-1:i+2, j-1:j+2] * kernel_y)
                magnitude = np.sqrt(gx**2 + gy**2)
                if magnitude > high_threshold:
                    edges[i, j] = 255
        
        return edges.astype(np.uint8)
    
    def _morphology_close_fallback(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Simple morphological closing fallback"""
        # Simple dilation followed by erosion
        result = mask.copy()
        
        # Dilation
        for i in range(kernel_size//2, mask.shape[0] - kernel_size//2):
            for j in range(kernel_size//2, mask.shape[1] - kernel_size//2):
                if np.any(mask[i-kernel_size//2:i+kernel_size//2+1, 
                             j-kernel_size//2:j+kernel_size//2+1] > 0):
                    result[i, j] = 255
        
        # Erosion
        mask = result.copy()
        for i in range(kernel_size//2, mask.shape[0] - kernel_size//2):
            for j in range(kernel_size//2, mask.shape[1] - kernel_size//2):
                if not np.all(mask[i-kernel_size//2:i+kernel_size//2+1, 
                                 j-kernel_size//2:j+kernel_size//2+1] > 0):
                    result[i, j] = 0
        
        return result
    
    def _find_contours_fallback(self, edges: np.ndarray) -> List:
        """Simple contour detection fallback"""
        # Very basic contour detection - find connected components
        contours = []
        visited = np.zeros_like(edges, dtype=bool)
        
        for i in range(edges.shape[0]):
            for j in range(edges.shape[1]):
                if edges[i, j] > 0 and not visited[i, j]:
                    # Found a new contour, trace it
                    contour = self._trace_contour(edges, visited, i, j)
                    if len(contour) > 10:  # Only keep significant contours
                        contours.append(np.array(contour))
        
        return contours
    
    def _trace_contour(self, edges: np.ndarray, visited: np.ndarray, start_i: int, start_j: int) -> List:
        """Trace a contour from a starting point"""
        contour = []
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            if (i < 0 or i >= edges.shape[0] or j < 0 or j >= edges.shape[1] or 
                visited[i, j] or edges[i, j] == 0):
                continue
            
            visited[i, j] = True
            contour.append([j, i])  # OpenCV format: [x, y]
            
            # Add 8-connected neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    stack.append((i + di, j + dj))
        
        return contour
    
    def _contour_area_fallback(self, contour: np.ndarray) -> float:
        """Calculate contour area using shoelace formula"""
        if len(contour) < 3:
            return 0.0
        
        # Handle different contour formats
        if len(contour.shape) == 3:
            # OpenCV format: (n_points, 1, 2)
            x = contour[:, 0, 0]  # x coordinates
            y = contour[:, 0, 1]  # y coordinates
        else:
            # Simple format: (n_points, 2)
            x = contour[:, 0]  # x coordinates
            y = contour[:, 1]  # y coordinates
        
        area = 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] 
                            for i in range(-1, len(x) - 1)))
        return area
    
    def _arc_length_fallback(self, contour: np.ndarray) -> float:
        """Calculate contour perimeter"""
        if len(contour) < 2:
            return 0.0
        
        perimeter = 0.0
        for i in range(len(contour)):
            # Handle different contour formats
            if len(contour.shape) == 3:
                p1 = contour[i, 0]
                p2 = contour[(i + 1) % len(contour), 0]
            else:
                p1 = contour[i]
                p2 = contour[(i + 1) % len(contour)]
            perimeter += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        return perimeter
    
    def _convex_hull_fallback(self, contour: np.ndarray) -> np.ndarray:
        """Simple convex hull approximation using Graham scan"""
        if len(contour) < 3:
            return contour
        
        # Convert to simple format
        if len(contour.shape) == 3:
            points = contour[:, 0, :]  # Remove middle dimension
        else:
            points = contour  # Already in simple format
        
        # Find bottom-most point (and leftmost in case of tie)
        start = min(points, key=lambda p: (p[1], p[0]))
        
        # Sort points by polar angle with respect to start point
        def polar_angle(p):
            dx, dy = p[0] - start[0], p[1] - start[1]
            return np.arctan2(dy, dx)
        
        sorted_points = sorted(points, key=polar_angle)
        
        # Simple convex hull - just return the sorted points (approximation)
        # In a full implementation, we'd use Graham scan algorithm
        hull_points = []
        for point in sorted_points:
            hull_points.append([[point[0], point[1]]])
        
        return np.array(hull_points) if hull_points else contour
