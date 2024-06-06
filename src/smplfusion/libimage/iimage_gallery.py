import os
from .utils import bytes2html

class ImageGallery:
    def __init__(self, images, captions = None, size='auto', max_rows = -1, root_path = '/', caption_font_size = '2em'):
        self.size = size
        self.images = images
        self.display_str = None
        self.captions = captions if captions is not None else [''] * len(self.images)
        self.max_rows = max_rows
        self.root_path = root_path
        self.caption_font_size = caption_font_size

    def generate_display(self):
        """Shows a set of images in a gallery that flexes with the width of the notebook.
        
        Parameters
        ----------
        images: list of str or bytes
            URLs or bytes of images to display

        row_height: str
            CSS height value to assign to all images. Set to 'auto' by default to show images
            with their native dimensions. Set to a value e.g. '250px' to make all rows
            in the gallery equal height.
        """
        figures = []
        row_figures = 0
        for image, caption in zip(self.images, self.captions):
            if isinstance(image, str):
                with open(image,'rb') as f:
                    link = os.path.relpath(image, self.root_path)
                    # src = bytes2html(f.read(), width = self.size)
                    src = bytes2html(f.read())
                    src = f'<a href="{link}" >{src}</a>'
            else:
                if image.display_str is None: image.generate_display()
                if image.is_video():
                    src = image.display_str
                else:
                    # src = _src_from_data(image.display_str)
                    src = image.to_html(width = "100%", root_path = self.root_path)
            if caption != '':
                caption = f'<figcaption style="position: absolute; top: 0; left: 0; width: 250px; font-size: {self.caption_font_size}; font-style: italic; text-align: center; color: white; background-color: rgba(0, 0, 0, 0.5); ">{caption}</figcaption>'
            figures.append(f'''
                <figure style="position: relative; margin: 5px !important; flex-basis: {self.size};">
                {caption}
                {src}
                </figure>
            ''')
            row_figures += 1
            if row_figures == self.max_rows: 
                row_figures = 0
                figures.append('<div style="flex-basis: 100%; height: 0;"></div>')
        self.display_str = f'''
            <div style="display: flex; flex-flow: row wrap; text-align: left;">
            {''.join(figures)}
            </div>
        '''
    
    def _repr_html_(self):
        if self.display_str is None: self.generate_display()
        return self.display_str
    
    def save(self, path):
        if self.display_str is None: self.generate_display()
        with open(path, 'w') as f:
            f.write(self.display_str)