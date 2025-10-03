
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf092-loop-switch-combo/cf092-loop-switch-combo_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_loop_switchv>:
100000360: 52800008    	mov	w8, #0x0                ; =0
100000364: 5280000e    	mov	w14, #0x0               ; =0
100000368: 5280014a    	mov	w10, #0xa               ; =10
10000036c: 12800009    	mov	w9, #-0x1               ; =-1
100000370: 5280002b    	mov	w11, #0x1               ; =1
100000374: 5295556c    	mov	w12, #0xaaab            ; =43691
100000378: 72b5554c    	movk	w12, #0xaaaa, lsl #16
10000037c: aa0e03ed    	mov	x13, x14
100000380: 9bac7d0e    	umull	x14, w8, w12
100000384: d361fdce    	lsr	x14, x14, #33
100000388: 0b0e05ce    	add	w14, w14, w14, lsl #1
10000038c: 4b0e016e    	sub	w14, w11, w14
100000390: 0b0e01ae    	add	w14, w13, w14
100000394: 11000508    	add	w8, w8, #0x1
100000398: 11000529    	add	w9, w9, #0x1
10000039c: 1100056b    	add	w11, w11, #0x1
1000003a0: 7100054a    	subs	w10, w10, #0x1
1000003a4: 54fffec1    	b.ne	0x10000037c <__Z16test_loop_switchv+0x1c>
1000003a8: 5295556a    	mov	w10, #0xaaab            ; =43691
1000003ac: 72b5554a    	movk	w10, #0xaaaa, lsl #16
1000003b0: 9baa7d29    	umull	x9, w9, w10
1000003b4: d361fd29    	lsr	x9, x9, #33
1000003b8: 0b090529    	add	w9, w9, w9, lsl #1
1000003bc: 4b090108    	sub	w8, w8, w9
1000003c0: 0b0d0100    	add	w0, w8, w13
1000003c4: d65f03c0    	ret

00000001000003c8 <_main>:
1000003c8: 52800008    	mov	w8, #0x0                ; =0
1000003cc: 5280000e    	mov	w14, #0x0               ; =0
1000003d0: 5280014a    	mov	w10, #0xa               ; =10
1000003d4: 12800009    	mov	w9, #-0x1               ; =-1
1000003d8: 5280002b    	mov	w11, #0x1               ; =1
1000003dc: 5295556d    	mov	w13, #0xaaab            ; =43691
1000003e0: 72b5554d    	movk	w13, #0xaaaa, lsl #16
1000003e4: aa0e03ec    	mov	x12, x14
1000003e8: 9bad7d0e    	umull	x14, w8, w13
1000003ec: d361fdce    	lsr	x14, x14, #33
1000003f0: 0b0e05ce    	add	w14, w14, w14, lsl #1
1000003f4: 4b0e016e    	sub	w14, w11, w14
1000003f8: 0b0e018e    	add	w14, w12, w14
1000003fc: 11000508    	add	w8, w8, #0x1
100000400: 11000529    	add	w9, w9, #0x1
100000404: 1100056b    	add	w11, w11, #0x1
100000408: 7100054a    	subs	w10, w10, #0x1
10000040c: 54fffec1    	b.ne	0x1000003e4 <_main+0x1c>
100000410: 5295556a    	mov	w10, #0xaaab            ; =43691
100000414: 72b5554a    	movk	w10, #0xaaaa, lsl #16
100000418: 9baa7d29    	umull	x9, w9, w10
10000041c: d361fd29    	lsr	x9, x9, #33
100000420: 0b090529    	add	w9, w9, w9, lsl #1
100000424: 4b090108    	sub	w8, w8, w9
100000428: 0b0c0100    	add	w0, w8, w12
10000042c: d65f03c0    	ret
