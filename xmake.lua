add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

-- Helper to prevent install
function no_install(target)
    -- Do nothing
end

-- [CHANGE] static -> object
target("llaisys-utils")
    set_kind("object")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/utils/*.cpp")
    on_install(no_install)
target_end()

-- [CHANGE] static -> object
target("llaisys-device")
    set_kind("object")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/device/*.cpp")
    on_install(no_install)
target_end()

-- [CHANGE] static -> object
target("llaisys-core")
    set_kind("object")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/core/*/*.cpp")
    on_install(no_install)
target_end()

-- [CHANGE] static -> object
target("llaisys-tensor")
    set_kind("object")
    add_deps("llaisys-core")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/tensor/*.cpp")
    on_install(no_install)
target_end()

-- [CHANGE] static -> object
target("llaisys-ops")
    set_kind("object")
    add_deps("llaisys-ops-cpu")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/ops/*/*.cpp")
    on_install(no_install)
target_end()

-- [CHANGE] static -> object
target("llaisys-models")
    set_kind("object")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    add_deps("llaisys-utils")
    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    add_files("src/llaisys/models/*.cpp")
    on_install(no_install)
target_end()

target("llaisys")
    set_kind("shared")
    -- Link all objects directly into the shared library
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")
    add_deps("llaisys-models")

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")
    set_installdir(".")
    
    after_install(function (target)
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
target_end()
