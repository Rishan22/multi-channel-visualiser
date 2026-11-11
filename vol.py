import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from pathlib import Path
import time
import gc

@dataclass
class RelationshipFrame:
    timestamp: float
    relationships: Dict[str, np.ndarray]
    view_position: Tuple[float, float]
    original_size: Tuple[int, int]
    
    def compress(self) -> None:
        h, w = self.original_size
        target_h, target_w = h // 4, w // 4
        
        for key in self.relationships:
            self.relationships[key] = cv2.resize(
                self.relationships[key], 
                (target_w, target_h), 
                interpolation=cv2.INTER_AREA
            ).astype(np.uint8)
    
    def decompress(self) -> None:
        h, w = self.original_size
        for key in self.relationships:
            self.relationships[key] = cv2.resize(
                self.relationships[key],
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )

class FrameBuffer:
    def __init__(self, max_frames: int = 30):
        self.max_frames = max_frames
        self.frames: List[RelationshipFrame] = []
    
    def add_frame(self, frame: RelationshipFrame) -> None:
        frame.compress()
        self.frames.append(frame)
        while len(self.frames) > self.max_frames:
            self.frames.pop(0)
        gc.collect()
    
    def get_frame(self, index: int, decompress: bool = True) -> RelationshipFrame:
        frame = self.frames[index]
        if decompress:
            frame.decompress()
        return frame

class PhotogrammetrySystem:
    def __init__(self, buffer_size: int = 30):
        self.modes = {
            'capture': 0,
            'view': 1,
            'relationships': 2,
            'wireframe': 3,
            'multiply': 4,     # New mode
            'ratio': 5,        # New mode
            'weighted': 6,     # New mode
            'hybrid': 7        # New mode
        }
        self.current_mode = 'capture'
        self.frame_buffer = FrameBuffer(max_frames=buffer_size)
        self.is_capturing = False
        self.view_angle = 0.0
    
    def extract_relationships(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        if frame is None:
            raise ValueError("Frame cannot be None")
            
        # Color space conversions
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0].copy()
        a_channel = lab[:,:,1].copy()
        b_channel = lab[:,:,2].copy()
        del lab
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:,:,0].copy()
        s_channel = hsv[:,:,1].copy()
        v_channel = hsv[:,:,2].copy()
        del hsv
        
        # Original relationships
        relationships = {
            'form_material': cv2.absdiff(l_channel, s_channel),
            'structure_depth': cv2.absdiff(l_channel, b_channel),
            'material_color': cv2.absdiff(s_channel, h_channel),
            
            # New multiplication relationships
            'volume_edges': cv2.multiply(l_channel, s_channel) / 255,
            'surface_continuity': cv2.multiply(l_channel, v_channel) / 255,
            
            # New ratio relationships
            'form_illumination': cv2.divide(l_channel, v_channel + 1, scale=255),
            'material_transition': cv2.divide(s_channel, l_channel + 1, scale=255),
            
            # New weighted relationships
            'form_detail': cv2.addWeighted(
                a_channel, 0.3,
                b_channel, 0.7,
                0
            ),
            'surface_depth': cv2.addWeighted(
                l_channel, 0.6,
                s_channel, 0.4,
                0
            )
        }
        
        # Normalize all relationships
        for key in relationships:
            relationships[key] = cv2.normalize(
                relationships[key],
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )
        
        return relationships
    
    def create_visualization(self, relationships: Dict[str, np.ndarray],
                           target_size: Tuple[int, int]) -> np.ndarray:
        # Ensure all relationships are at target size
        scaled_relationships = {
            key: cv2.resize(arr, (target_size[1], target_size[0]))
            for key, arr in relationships.items()
        }
        
        if self.current_mode == 'relationships':
            return cv2.merge([
                scaled_relationships['form_material'],
                scaled_relationships['structure_depth'],
                scaled_relationships['material_color']
            ])
            
        elif self.current_mode == 'wireframe':
            structure = cv2.addWeighted(
                scaled_relationships['form_material'],
                0.5,
                scaled_relationships['structure_depth'],
                0.5,
                0
            )
            return cv2.cvtColor(structure, cv2.COLOR_GRAY2BGR)
            
        elif self.current_mode == 'multiply':
            return cv2.merge([
                scaled_relationships['volume_edges'],
                scaled_relationships['surface_continuity'],
                scaled_relationships['material_color']
            ])
            
        elif self.current_mode == 'ratio':
            return cv2.merge([
                scaled_relationships['form_illumination'],
                scaled_relationships['material_transition'],
                scaled_relationships['structure_depth']
            ])
            
        elif self.current_mode == 'weighted':
            return cv2.merge([
                scaled_relationships['form_detail'],
                scaled_relationships['surface_depth'],
                scaled_relationships['material_color']
            ])
            
        elif self.current_mode == 'hybrid':
            # Combine multiple relationship types
            volume = cv2.addWeighted(
                scaled_relationships['volume_edges'],
                0.5,
                scaled_relationships['form_illumination'],
                0.5,
                0
            )
            surface = cv2.addWeighted(
                scaled_relationships['surface_continuity'],
                0.5,
                scaled_relationships['surface_depth'],
                0.5,
                0
            )
            return cv2.merge([volume, surface, scaled_relationships['material_color']])
            
        else:  # capture or view mode
            base = cv2.merge([
                scaled_relationships['form_material'],
                scaled_relationships['structure_depth'],
                scaled_relationships['material_color']
            ])
            structure = cv2.addWeighted(
                scaled_relationships['form_material'],
                0.5,
                scaled_relationships['structure_depth'],
                0.5,
                0
            )
            return cv2.addWeighted(
                base,
                0.7,
                cv2.cvtColor(structure, cv2.COLOR_GRAY2BGR),
                0.3,
                0
            )
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None:
            raise ValueError("Frame cannot be None")
            
        h, w = frame.shape[:2]
        current_size = (h, w)
        
        relationships = self.extract_relationships(frame)
        
        if self.is_capturing:
            prev_relationships = None
            if self.frame_buffer.frames:
                prev_frame = self.frame_buffer.get_frame(-1, decompress=True)
                prev_relationships = prev_frame.relationships
            
            position = self.calculate_position_change(
                relationships,
                prev_relationships,
                current_size
            ) if prev_relationships else (0.0, 0.0)
            
            new_frame = RelationshipFrame(
                timestamp=time.time(),
                relationships=relationships,
                view_position=position,
                original_size=current_size
            )
            self.frame_buffer.add_frame(new_frame)
        
        visualization = self.create_visualization(relationships, current_size)
        
        try:
            # Changed from hconcat to vconcat for vertical stacking
            display = cv2.vconcat([visualization, frame])
            del visualization
            return display
        except cv2.error:
            return visualization
    
    def calculate_position_change(self, current: Dict[str, np.ndarray],
                                previous: Dict[str, np.ndarray],
                                current_size: Tuple[int, int]) -> Tuple[float, float]:
        if previous is None:
            return (0.0, 0.0)
        
        prev_size = next(iter(previous.values())).shape[:2]
        if prev_size != current_size:
            previous = {
                key: cv2.resize(arr, (current_size[1], current_size[0]))
                for key, arr in previous.items()
            }
        
        dx = np.mean(cv2.absdiff(current['form_material'], previous['form_material']))
        dy = np.mean(cv2.absdiff(current['structure_depth'], previous['structure_depth']))
        
        return (float(dx), float(dy))
    
    def run(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
            
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                display = self.process_frame(frame.copy())
                cv2.imshow('Photogrammetry System', display)
                
                del frame
                gc.collect()
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.is_capturing = not self.is_capturing
                    print(f"Capturing: {self.is_capturing}")
                elif key == ord('m'):
                    modes = list(self.modes.keys())
                    current_idx = modes.index(self.current_mode)
                    self.current_mode = modes[(current_idx + 1) % len(modes)]
                    print(f"Switched to mode: {self.current_mode}")
                elif key == ord('s'):
                    self.save_capture('capture_data/capture.json')
                    print("Saved capture data")
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
            gc.collect()

    def save_capture(self, filename: str) -> None:
        data = {
            'frames': [
                {
                    'timestamp': frame.timestamp,
                    'view_position': frame.view_position,
                    'original_size': frame.original_size
                }
                for frame in self.frame_buffer.frames
            ]
        }
        
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(data, f)

def main():
    system = PhotogrammetrySystem(buffer_size=30)
    system.run()

if __name__ == "__main__":
    main()