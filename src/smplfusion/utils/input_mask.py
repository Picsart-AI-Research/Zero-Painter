import torch 
from ..libimage import IImage

class InputMask:
    def to(self, device): return InputMask(self.image, device = device)
    def cuda(self): return InputMask(self.image, device = 'cuda')
    def cpu(self): return InputMask(self.image, device = 'cpu')

    def __init__(self, input_image, device = 'cpu'):
        '''
        args:
            input_image: (b,c,h,w) tensor
        '''
        if hasattr(input_image, 'is_iimage'):
            self.image = input_image
            self.val512 = self.full = (input_image.torch(0) > 0.5).float()
        elif isinstance(input_image, torch.Tensor):
            self.val512 = self.full = input_image.clone()
            self.image = IImage(input_image,0)
        
        self.h,self.w = h,w = self.val512.shape[-2:]
        self.shape   = [self.h, self.w]
        self.shape64 = [self.h // 8, self.w // 8]
        self.shape32 = [self.h // 16, self.w // 16]
        self.shape16 = [self.h // 32, self.w // 32]
        self.shape8  = [self.h // 64, self.w // 64]

        self.res   = self.h * self.w
        self.res64 = self.res // 64
        self.res32 = self.res // 64 // 4
        self.res16 = self.res // 64 // 16
        self.res8  = self.res // 64 // 64

        self.img = self.image
        self.img512 = self.image
        self.img64 = self.image.resize((h//8,w//8))
        self.img32 = self.image.resize((h//16,w//16))
        self.img16 = self.image.resize((h//32,w//32))
        self.img8  = self.image.resize((h//64,w//64))

        self.val64 = (self.img64.torch(0) > 0.5).float()
        self.val32 = (self.img32.torch(0) > 0.5).float()
        self.val16 = (self.img16.torch(0) > 0.5).float()
        self.val8  = ( self.img8.torch(0) > 0.5).float()

    
    def get_res(self, q, device = 'cpu'):
        if q.shape[1] == self.res64: return self.val64.to(device)
        if q.shape[1] == self.res32: return self.val32.to(device)
        if q.shape[1] == self.res16: return self.val16.to(device)
        if q.shape[1] == self.res8: return self.val8.to(device)

    def _repr_png_(self):
        return self.img16._repr_png_()
    
    def get_res(self, q, device = 'cpu'):
        if q.shape[1] == self.res64: return self.val64.to(device)
        if q.shape[1] == self.res32: return self.val32.to(device)
        if q.shape[1] == self.res16: return self.val16.to(device)
        if q.shape[1] == self.res8: return self.val8.to(device)

    def get_shape(self, q, device = 'cpu'):
        if q.shape[1] == self.res64: return self.shape64
        if q.shape[1] == self.res32: return self.shape32
        if q.shape[1] == self.res16: return self.shape16
        if q.shape[1] == self.res8: return self.shape8

    def get_res_val(self, q, device = 'cpu'):
        if q.shape[1] == self.res64: return 64
        if q.shape[1] == self.res32: return 32
        if q.shape[1] == self.res16: return 16
        if q.shape[1] == self.res8: return 8


# class InputMask2:
#     def to(self, device): return InputMask2(self.image, device = device)
#     def cuda(self): return InputMask2(self.image, device = 'cuda')
#     def cpu(self): return InputMask2(self.image, device = 'cpu')

#     def __init__(self, input_image, device = 'cpu'):
#         '''
#         args:
#             input_image: (b,c,h,w) tensor
#         '''
#         if hasattr(input_image, 'is_iimage'):
#             self.image = input_image
#             self.val512 = self.full = input_image.torch(0).bool().float()
#         elif isinstance(input_image, torch.Tensor):
#             self.val512 = self.full = input_image.clone()
#             self.image = IImage(input_image,0)
        
#         self.h,self.w = h,w = self.val512.shape[-2:]
#         self.shape   = [self.h, self.w]
#         self.shape64 = [self.h // 8, self.w // 8]
#         self.shape32 = [self.h // 16, self.w // 16]
#         self.shape16 = [self.h // 32, self.w // 32]
#         self.shape8  = [self.h // 64, self.w // 64]

#         self.res   = self.h * self.w
#         self.res64 = self.res // 64
#         self.res32 = self.res // 64 // 4
#         self.res16 = self.res // 64 // 16
#         self.res8  = self.res // 64 // 64

#         self.img = self.image
#         self.img512 = self.image
#         self.img64 = self.image.resize((h//8,w//8))
#         self.img32 = self.image.resize((h//16,w//16))
#         self.img16 = self.image.resize((h//32,w//32)).dilate(1)
#         self.img8  = self.image.resize((h//64,w//64)).dilate(1)

#         self.val64 = self.img64.torch(0).bool().float()
#         self.val32 = self.img32.torch(0).bool().float()
#         self.val16 = self.img16.torch(0).bool().float()
#         self.val8  =  self.img8.torch(0).bool().float()

    
#     def get_res(self, q, device = 'cpu'):
#         if q.shape[1] == self.res64: return self.val64.to(device)
#         if q.shape[1] == self.res32: return self.val32.to(device)
#         if q.shape[1] == self.res16: return self.val16.to(device)
#         if q.shape[1] == self.res8: return self.val8.to(device)

#     def _repr_png_(self):
#         return self.img16._repr_png_()
    
#     def get_res(self, q, device = 'cpu'):
#         if q.shape[1] == self.res64: return self.val64.to(device)
#         if q.shape[1] == self.res32: return self.val32.to(device)
#         if q.shape[1] == self.res16: return self.val16.to(device)
#         if q.shape[1] == self.res8: return self.val8.to(device)

#     def get_shape(self, q, device = 'cpu'):
#         if q.shape[1] == self.res64: return self.shape64
#         if q.shape[1] == self.res32: return self.shape32
#         if q.shape[1] == self.res16: return self.shape16
#         if q.shape[1] == self.res8: return self.shape8

#     def get_res_val(self, q, device = 'cpu'):
#         if q.shape[1] == self.res64: return 64
#         if q.shape[1] == self.res32: return 32
#         if q.shape[1] == self.res16: return 16
#         if q.shape[1] == self.res8: return 8