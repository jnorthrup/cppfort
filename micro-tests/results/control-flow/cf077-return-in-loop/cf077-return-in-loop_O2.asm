
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf077-return-in-loop/cf077-return-in-loop_O2.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z19test_return_in_loopi>:
100000360: aa0003e8    	mov	x8, x0
100000364: 52800c69    	mov	w9, #0x63               ; =99
100000368: 71018c1f    	cmp	w0, #0x63
10000036c: 1a893000    	csel	w0, w0, w9, lo
100000370: 7100111f    	cmp	w8, #0x4
100000374: 54000062    	b.hs	0x100000380 <__Z19test_return_in_loopi+0x20>
100000378: 52800009    	mov	w9, #0x0                ; =0
10000037c: 1400000b    	b	0x1000003a8 <__Z19test_return_in_loopi+0x48>
100000380: 71018d1f    	cmp	w8, #0x63
100000384: 1a893109    	csel	w9, w8, w9, lo
100000388: 11000529    	add	w9, w9, #0x1
10000038c: 7200052a    	ands	w10, w9, #0x3
100000390: 5280008b    	mov	w11, #0x4               ; =4
100000394: 1a8a016a    	csel	w10, w11, w10, eq
100000398: 4b0a0129    	sub	w9, w9, w10
10000039c: aa0903ea    	mov	x10, x9
1000003a0: 7100114a    	subs	w10, w10, #0x4
1000003a4: 54ffffe1    	b.ne	0x1000003a0 <__Z19test_return_in_loopi+0x40>
1000003a8: 4b090108    	sub	w8, w8, w9
1000003ac: 51019129    	sub	w9, w9, #0x64
1000003b0: 340000a8    	cbz	w8, 0x1000003c4 <__Z19test_return_in_loopi+0x64>
1000003b4: 51000508    	sub	w8, w8, #0x1
1000003b8: 31000529    	adds	w9, w9, #0x1
1000003bc: 54ffffa3    	b.lo	0x1000003b0 <__Z19test_return_in_loopi+0x50>
1000003c0: 12800000    	mov	w0, #-0x1               ; =-1
1000003c4: d65f03c0    	ret

00000001000003c8 <_main>:
1000003c8: 52800540    	mov	w0, #0x2a               ; =42
1000003cc: d65f03c0    	ret
