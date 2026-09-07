/**
 * hazards.js — what each storm morphology typically threatens.
 *
 * Plain classic script (popup.js is not a module), so this just defines a global.
 * Keyed by the CLASS_NAMES values in inference.js.
 *
 * severity drives the colour of the dot only. It reflects the hazard's TYPICAL
 * prominence for that storm type — not a forecast for the specific cell on screen.
 * Nothing here is a substitute for an actual NWS warning.
 */

const HAZARDS = {
  supercell: {
    headline: 'Rotating storm — the highest-end hazard set',
    items: [
      {
        severity: 'high',
        name: 'Tornadoes',
        detail: 'A mesocyclone — and more importantly a tight velocity couplet — means a ' +
                'tornado may be present or developing. Risk increases when the supercell ' +
                'is isolated.'
      },
      {
        severity: 'high',
        name: 'Large hail',
        detail: 'Reflectivity above roughly 65 dBZ suggests 2"+ stones are possible, ' +
                'though hail is typically smaller.'
      },
      {
        severity: 'moderate',
        name: 'Damaging winds',
        detail: 'The rear-flank downdraft, especially alongside a strong mesocyclone, ' +
                'can produce damaging gusts.'
      },
      {
        severity: 'moderate',
        name: 'Frequent lightning',
        detail: 'High cloud-to-ground flash rates near the updraft.'
      },
      {
        severity: 'low',
        name: 'Flash flooding',
        detail: 'If the cell is slow-moving or repeatedly crosses the same area.'
      }
    ],
    note: 'Long-lived (often hours) and frequently right-moving relative to the mean flow. Discrete cells like this warrant the closest attention.'
  },

  bowecho_squall: {
    headline: 'Wind-driven line segment — damage risk is mostly wind',
    items: [
      {
        severity: 'high',
        name: 'Damaging straight-line winds',
        detail: 'The defining hazard. Strongest at the bow apex where the rear-inflow jet ' +
                'reaches the surface, and swaths can run for hundreds of miles — a ' +
                'long-lived event may qualify as a derecho.'
      },
      {
        severity: 'moderate',
        name: 'Heavy rain',
        detail: 'Brief but intense as the line passes; localised flooding is possible.'
      },
      {
        severity: 'low',
        name: 'Brief spin-up tornadoes',
        detail: 'Mesovortex spin-ups are possible but typically weak, with very little ' +
                'warning lead time.'
      },
      {
        severity: 'low',
        name: 'Hail',
        detail: 'Usually small to marginally severe; not the main threat.'
      }
    ],
    note: 'The apex of the bow is where the strongest winds concentrate. Damage tends to be broad rather than localised.'
  },

  qlcs_squall: {
    headline: 'Linear convective system — continuous leading-edge threat',
    items: [
      {
        severity: 'high',
        name: 'Damaging winds',
        detail: 'A near-continuous swath along the leading edge as the line moves through.'
      },
      {
        severity: 'high',
        name: 'QLCS tornadoes',
        detail: 'Brief, fast-moving, often nocturnal, and notoriously short on warning lead time.'
      },
      {
        severity: 'moderate',
        name: 'Flash flooding',
        detail: 'Especially if the line slows or cells repeatedly track over the same ground.'
      },
      {
        severity: 'moderate',
        name: 'Frequent lightning',
        detail: 'Continuous along the convective line.'
      },
      {
        severity: 'low',
        name: 'Small hail',
        detail: 'Common but usually below severe limits.'
      }
    ],
    note: 'Tornadoes embedded in a line are hard to see and hard to warn on. Treat the whole leading edge as the threat, not one point.'
  },

  multicell: {
    headline: 'Cluster of cells — hazards move as cells form and decay',
    items: [
      {
        severity: 'moderate',
        name: 'Flash flooding',
        detail: 'The main sustained risk when cells repeatedly train over the same area.'
      },
      {
        severity: 'low',
        name: 'Pulse severe hail and wind',
        detail: 'Brief bursts as individual cells reach peak intensity, then collapse.'
      },
      {
        severity: 'low',
        name: 'Frequent lightning',
        detail: 'Across the whole cluster, including between active cells.'
      }
    ],
    note: 'Threat shifts around within the cluster rather than staying with one cell. Watch which cells are intensifying.'
  },

  single_cell: {
    headline: 'Isolated cell — usually short-lived and sub-severe',
    items: [
      {
        severity: 'moderate',
        name: 'Lightning',
        detail: 'The primary hazard. Strikes can reach well outside the rain area.'
      },
      {
        severity: 'moderate',
        name: 'Brief heavy rain',
        detail: 'Localised ponding and poor visibility for a short period.'
      },
      {
        severity: 'low',
        name: 'Small hail',
        detail: 'Possible near the core; severe sizes are uncommon.'
      },
      {
        severity: 'low',
        name: 'Wind',
        detail: 'A brief damaging gust is possible as the cell collapses.'
      }
    ],
    note: 'Typically lasts 20–40 minutes. Rarely severe, but pulse cells can produce one brief burst near peak intensity.'
  },

  nonradar_image: {
    headline: 'No radar detected in this selection',
    items: [
      {
        severity: 'none',
        name: 'Nothing to assess',
        detail: 'The crop does not look like radar reflectivity imagery.'
      }
    ],
    note: 'Try selecting a tighter box around the radar display itself, avoiding page background, legends and menus.'
  }
};
