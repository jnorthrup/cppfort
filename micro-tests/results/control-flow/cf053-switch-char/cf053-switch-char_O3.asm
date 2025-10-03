
/Users/jim/work/cppfort/micro-tests/results/control-flow/cf053-switch-char/cf053-switch-char_O3.out:	file format mach-o arm64

Disassembly of section __TEXT,__text:

0000000100000360 <__Z16test_switch_charc>:
100000360: 51018408    	sub	w8, w0, #0x61
100000364: 12001d09    	and	w9, w8, #0xff
100000368: 71000d1f    	cmp	w8, #0x3
10000036c: 1a8927e0    	csinc	w0, wzr, w9, hs
100000370: d65f03c0    	ret

0000000100000374 <_main>:
100000374: 52800040    	mov	w0, #0x2                ; =2
100000378: d65f03c0    	ret
