
/Users/jim/work/cppfort/micro-tests/results/memory/mem099-array-find/mem099-array-find_O1.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

00000001000003b0 <__Z15test_array_findi>:
1000003b0: aa0003e8    	mov	x8, x0
1000003b4: d2800000    	mov	x0, #0x0                ; =0
1000003b8: 90000009    	adrp	x9, 0x100000000
1000003bc: 910fa129    	add	x9, x9, #0x3e8
1000003c0: b860792a    	ldr	w10, [x9, x0, lsl #2]
1000003c4: 6b08015f    	cmp	w10, w8
1000003c8: 540000a0    	b.eq	0x1000003dc <__Z15test_array_findi+0x2c>
1000003cc: 91000400    	add	x0, x0, #0x1
1000003d0: f100141f    	cmp	x0, #0x5
1000003d4: 54ffff61    	b.ne	0x1000003c0 <__Z15test_array_findi+0x10>
1000003d8: 12800000    	mov	w0, #-0x1               ; =-1
1000003dc: d65f03c0    	ret

00000001000003e0 <_main>:
1000003e0: 52800040    	mov	w0, #0x2                ; =2
1000003e4: d65f03c0    	ret
